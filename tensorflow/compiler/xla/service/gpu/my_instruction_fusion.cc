/*
 * my_instruction_fusion.cc
 *
 *  Created on: Jan 21, 2021
 *      Author: admin
 */



/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/xla/service/gpu/my_instruction_fusion.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/fusion_node_indexing_evaluation.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_fusible.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/fusion_queue.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

#include <chrono>

using namespace std;
using namespace chrono;

namespace xla {
namespace gpu {

namespace {
bool ElementIsF32OrF16(const Shape& shape) {
  PrimitiveType type = shape.element_type();
  return type == F32 || type == F16;
}

size_t WriteCallback(char *contents, size_t size, size_t nmemb, void *userp)
{
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}
float GetEstimation(HloModule* module_) {
	CURL *curl;
	CURLcode res;

	/* In windows, this will init the winsock stuff */
	curl_global_init(CURL_GLOBAL_ALL);

	/* get a curl handle */
	curl = curl_easy_init();
	std::string readBuffer;

	if(curl) {
	 /* First set the URL that is about to receive our POST. This URL can
		just as well be a https:// URL if that is what should receive the
		data. */
		curl_easy_setopt(curl, CURLOPT_URL, "http://net-g12:3335/predict");

		struct curl_slist *list = NULL;
		list = curl_slist_append(list, "Content-Type: application/x-protobuf");


		curl_easy_setopt(curl, CURLOPT_HTTPHEADER, list);

		/* Now specify the POST data */
		std::string send_str;
		auto my_hlo_proto = absl::make_unique<HloProto>();
		*my_hlo_proto->mutable_hlo_module() = module_->ToProto();
		my_hlo_proto->SerializeToString(&send_str);
		curl_easy_setopt(curl, CURLOPT_POSTFIELDS, (void*)(send_str.c_str()));
		int size = my_hlo_proto->ByteSizeLong();

		curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE, size);

		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);


		/* Perform the request, res will get the return code */
		res = curl_easy_perform(curl);
		/* Check for errors */
		if(res != CURLE_OK)
		fprintf(stderr, "curl_easy_perform() failed: %s\n",
			   curl_easy_strerror(res));

		/* always cleanup */
		curl_slist_free_all(list);
		curl_easy_cleanup(curl);
		}
		curl_global_cleanup();
		return std::stof(readBuffer);
}






}  // namespace

/* My implenation for custimized op fusion
 *
 */

/*static*/ bool MyGpuInstructionFusion::IsExpensive(
    const HloInstruction& instruction) {
  // We say that some floating-point math ops are cheap on the GPU. Unlike other
  // intrinsics that can be expanded into many instructions, Div and Rsqrt are
  // lowered into single hardware instructions.
  switch (instruction.opcode()) {
    case HloOpcode::kDivide:
    case HloOpcode::kRsqrt:
      if (ElementIsF32OrF16(instruction.shape())) {
        return false;
      }
      break;
    default:
      break;
  }
  return InstructionFusion::IsExpensive(instruction);
}



std::unique_ptr<FusionQueue> MyGpuInstructionFusion::GetRandomFusionQueue(
    HloComputation* computation){


	/*auto res = randomFusionQueue_map_.find(computation);
	if (res==randomFusionQueue_map_.end()){

		auto emplace_res = randomFusionQueue_map_.emplace(computation,RandomFusionQueue(computation));
		res = emplace_res.first;
	}

	return &(res->second);*/

	return absl::make_unique<RandomFusionQueue>(computation);


}

bool MyGpuInstructionFusion::ShouldFuseInexpensiveChecks(HloInstruction* consumer,
                                                       int64 operand_index) {
  HloInstruction* producer = consumer->mutable_operand(operand_index);

  // Output fusions are not currently supported on GPUs.
  if (producer->opcode() == HloOpcode::kFusion) {
    VLOG(4) << "Producer " << producer->name() << " is a fusion op";
    //if (std::getenv("PRINT"))std::cout << "Producer " << producer->name() << " is a fusion op"<<std::endl;

    return false;
  }
  // Cost condition: not fuse (simple, expensive producers) and (consumers who
  // reuse operand elements).
  if (producer->opcode() != HloOpcode::kFusion && is_expensive(*producer) &&
      ReusesOperandElements(consumer, operand_index)) {
    VLOG(4) << "Do not fuse simple, expensive producer " << producer->name()
            << " and consumer which reuses operand elements.";
    //if (std::getenv("PRINT"))std::cout << "Do not fuse simple, expensive producer " << producer->name()<< " and consumer which reuses operand elements."<<std::endl;
    return false;
  }

  if (!IsProducerConsumerFusible(*producer, *consumer) ||
      !InstructionFusion::ShouldFuse(consumer, operand_index)) {
    VLOG(4) << "Producer " << producer->name()
            << " is not fusible or should not be fused.";
    //if (std::getenv("PRINT"))std::cout << "Producer " << producer->name()<< " is not fusible or should not be fused."<<std::endl;
    return false;
  }
  return true;
}

bool MyGpuInstructionFusion::ShouldFuse(HloInstruction* consumer,
                                      int64 operand_index) {
  if (!ShouldFuseInexpensiveChecks(consumer, operand_index)) {
    VLOG(5) << "Not fusing inexpensive checks of operand " << operand_index
            << " of " << consumer->ToString();
    //if (std::getenv("PRINT"))std::cout << "Not fusing inexpensive checks of operand " << operand_index<< " of " << consumer->ToString()<<std::endl;
    return false;
  }
  auto producer = consumer->operand(operand_index);

  // The following checks are potentially expensive.
  if (FusionWouldBeTooLarge(*consumer, *producer,
                            /*is_consumer_producer_fusion=*/true)) {
    VLOG(5) << "Fusion of (" << producer->ToString() << ") into ("
            << consumer->ToString() << ") would be too large";
    //if (std::getenv("PRINT"))std::cout << "Fusion of (" << producer->ToString() << ") into ("<< consumer->ToString() << ") would be too large"<<std::endl;
    return false;
  }
  if (consumer->opcode() != HloOpcode::kFusion) {
    return true;
  }
  // Also check that our emitter can handle the fusion node. We currently can
  // have exponential time/memory requirements for emitting certain fusion
  // kernels, in which case we don't want to fuse.
  // TODO(b/119692968): Remove this once we have fixed our fusion emitter.
  if (fusion_node_evaluations_.find(consumer) ==
      fusion_node_evaluations_.end()) {
    // We have no cached results for this fusion node yet. This can happen when
    // we run the InstructionFusion pass more than once. We can only cache the
    // results within one run.
    fusion_node_evaluations_.emplace(consumer,
                                     FusionNodeIndexingEvaluation(consumer));
  }
  if (fusion_node_evaluations_.at(consumer).CodeDuplicationTooHigh(producer)) {
    VLOG(5) << "Fusion of " << producer->name() << " into " << consumer->name()
            << " would result in overly large code duplication.";
    //if (std::getenv("PRINT"))std::cout << "Fusion of " << producer->name() << " into " << consumer->name()<< " would result in overly large code duplication."<<std::endl;
    return false;
  }
  return true;
}

bool MyGpuInstructionFusion::ShouldFuseIntoMultiOutput(HloInstruction* consumer,
                                                     int64 operand_index) {
  return false;
}

HloInstruction::FusionKind MyGpuInstructionFusion::ChooseKind(
    const HloInstruction* producer, const HloInstruction* consumer) {
  return ChooseFusionKind(*producer, *consumer);
}

HloInstruction* MyGpuInstructionFusion::FuseInstruction(
    HloInstruction* fusion_instruction, HloInstruction* producer) {
  auto evaluation = fusion_node_evaluations_.find(fusion_instruction);
  if (evaluation == fusion_node_evaluations_.end()) {
    evaluation = fusion_node_evaluations_
                     .emplace(fusion_instruction,
                              FusionNodeIndexingEvaluation(fusion_instruction))
                     .first;
  }
  auto indexing_users = evaluation->second.RemoveFusionOperand(producer);
  HloInstruction* new_producer =
		InstructionFusion::FuseInstruction(fusion_instruction, producer);
  evaluation->second.UpdateEvaluationCache(new_producer, indexing_users);
  return new_producer;
}


StatusOr<HloInstruction*> MyGpuInstructionFusion::FuseSpecificInstruction(HloInstruction* instruction,FusionQueue*fusion_queue){
	if (std::getenv("PRINT"))std::cout<<"in FuseSpecificInstruction"<<std::endl;
    if (instruction == nullptr) {
    	if (std::getenv("PRINT"))std::cout<<"OUT FuseSpecificInstruction"<<std::endl;

      return nullptr;
    }
    if (!instruction->IsFusible() &&
        instruction->opcode() != HloOpcode::kFusion) {
    	if (std::getenv("PRINT"))std::cout<<"OUT FuseSpecificInstruction"<<std::endl;

        //if (std::getenv("PRINT"))if (std::getenv("PRINT"))std::cout << "instruction (" << instruction->name() << ") is not fusible"<<std::endl;
      return nullptr;
    }

    //auto fusion_queue =GetRandomFusionQueue(computation_);

    HloInstruction* fusion_instruction = nullptr;
	reachability_ = HloReachabilityMap::Build(computation_);

    auto do_not_duplicate = ComputeGloballyUnfusible(computation_->MakeInstructionPostOrder());

    auto sorted_operand_numbers =static_cast<RandomFusionQueue*>(fusion_queue)->GetSortedOperandNumbers(instruction);

    //if (std::getenv("PRINT"))if (std::getenv("PRINT"))std::cout<<"sorted_operand_numbers length:"<<sorted_operand_numbers.size()<<std::endl;

    for (int64 i : sorted_operand_numbers) {
      HloInstruction* operand = instruction->mutable_operand(i);
      VLOG(5) << "Considering fusion of: " << instruction->ToString()
              << " with operand " << operand->name();

      if (!operand->IsFusible()) {
      	//if (std::getenv("PRINT"))if (std::getenv("PRINT"))std::cout << "Operand (" << operand->name() << ") is not fusible"<<std::endl;
        continue;
      }

      // Consumes a unit of compiler fuel and returns true if we should
      // continue with the transformation.
      auto consume_fuel = [&] {
        return ConsumeFuel(name(), /*ran_out_of_fuel_msg=*/[&] {
          return absl::StrFormat("Not fusing operand %d of %s, namely, %s", i,
                                 instruction->ToString(),
                                 operand->ToString());
        });
      };

      // Try "regular" fusion if the operand may be duplicated. Otherwise,
      // perform multi-output fusion, unless this creates a cycle.
      //if (do_not_duplicate.count(operand)!=0){
      	//if (std::getenv("PRINT"))std::cout << "do_not_duplicate has " << operand->name()<<std::endl;
      //}
      //if (!ShouldFuse(instruction, i)){
      	//if (std::getenv("PRINT"))std::cout << "The " <<i<<"operand:"<<operand->name()<<"cannot fused to "<<"instruction: "<<instruction->name()<<std::endl;
      //}
      if (may_duplicate_&&do_not_duplicate.count(operand) == 0 &&
          ShouldFuse(instruction, i)) {
        if (consume_fuel()) {
          fusion_queue->PreFusion(operand, instruction);
          fusion_instruction = Fuse(operand, instruction);
        }
      }

      if (fusion_instruction == nullptr) {
        continue;
      }

      fusion_queue->OnFusingInstruction(fusion_instruction, operand,
                                        instruction);

      if (operand->user_count() == 0) {
        do_not_duplicate.erase(operand);
        // Operand is now dead. Remove from queue.
        fusion_queue->RemoveInstruction(operand);
        // Remove from computation.
        TF_RETURN_IF_ERROR(computation_->RemoveInstruction(operand));
      }

      if (fusion_instruction != instruction) {
        do_not_duplicate.erase(instruction);
      }
      break;
    }
	if (std::getenv("PRINT"))std::cout<<"OUT FuseSpecificInstruction"<<std::endl;

	reachability_.reset();

    return fusion_instruction;

}

StatusOr<bool> MyGpuInstructionFusion::RandomFuseOnce(){
	if (std::getenv("PRINT"))std::cout<<"    in RandomFuseOnce"<<std::endl;
	bool changed = false;

	while(!changed &&computation_list_.size()>0){
		int fuse_count = 0;
		int random_number= std::experimental::randint(0, int(computation_list_.size())-1);
		computation_ = computation_list_.at(random_number);

	    //reachability_ = reachability_map_.find(computation_)->second;
		//reachability_ = HloReachabilityMap::Build(computation_);
	    // If we allow duplications, we need to compute which instructions we do not
	    // want to duplicate based on a global analysis of the graph.

	    //do_not_duplicate =&(do_not_duplicate_map_.find(computation_)->second);
	    auto fusion_queue = GetRandomFusionQueue(computation_);

	    while (!changed){

	    	auto next_entry =
	    	  fusion_queue->DequeueNextInstructionAndOperandsToFuseInOrder();

	    	HloInstruction* instruction = next_entry.first;
	    	if (instruction == nullptr) {

	    		computation_list_.erase(computation_list_.begin()+random_number);
	    		//if (std::getenv("PRINT"))std::cout<<"fusion_queue nullptr"<<std::endl;
	    		break;
	    	}

	    	int random_times= std::experimental::randint(1,10);
	    	for (int i=0;i<random_times;i++){

	    		if (i!=0){
	    			//dequeue just fused instruction.
	    			auto fused_entry =static_cast<RandomFusionQueue*>(fusion_queue.get())->DequeueLastInstructionAndOperandsToFuseInOrder();
	    			auto fused_instruction = fused_entry.first;
	    			assert(fused_instruction==instruction);
	    		}

	    		TF_ASSIGN_OR_RETURN(instruction, FuseSpecificInstruction(instruction,fusion_queue.get()));
	    		if (instruction == nullptr) {
	    			//if (std::getenv("PRINT"))std::cout<<"FuseSpecificInstruction nullptr"<<std::endl;
	    			break;
	    		}else{
	    			changed = true;
	    		    ++fuse_count;

	    		}

	    	}

	    }

	}



	//if (std::getenv("PRINT"))std::cout << "Fusion count: " << fuse_count<<std::endl;
	if (std::getenv("PRINT"))std::cout<<"    out RandomFuseOnce"<<std::endl;

	return changed;


}

std::vector<HloComputation*>* MyGpuInstructionFusion::GetComputeLists(HloModule* module){


	auto res = module_computation_list_.find(module);
	if (res==module_computation_list_.end()){
		std::vector<HloComputation*> list = module_->MakeNonfusionComputationsSorted();
		auto emplace_res = module_computation_list_.emplace(module,list);
		res = emplace_res.first;
	}

	return &(res->second);

	//return std::make_shared<RandomFusionQueue>(computation);



}

StatusOr<bool> MyGpuInstructionFusion::Run(HloModule* module){


	auto estimate = GetEstimation(module);
	best_module_ = module->Clone(module->config(),"");
	best_estimation_ = estimate;



	std::unique_ptr<HloModule> cloned_module = module->Clone(module->config(),"");
	sampled_modules_.emplace(estimate,std::move(cloned_module));

	int sample_times =0;
	while(!sampled_modules_.empty()||sample_times<500){
		sample_times+=1;
		std::cout<<"sample_times: "<<sample_times<<std::endl;
		if (std::getenv("PRINT"))std::cout<<"Begin While"<<std::endl;
		std::cout<<"  Pop the first module from priority queue"<<std::endl;
		std::cout<<"  queue size:"<<sampled_modules_.size()<<std::endl;

		auto pop_item = sampled_modules_.begin();
		std::unique_ptr<HloModule> unique_pop_module = std::move(pop_item->second);
		auto pop_module = unique_pop_module.get();
		auto pop_estimate = pop_item->first;
		sampled_modules_.erase(pop_item);


		//TF_RETURN_IF_ERROR(InstructionFusion::Run(module).status());
		//TF_RETURN_IF_ERROR(multi_output_fuser_.Run(module).status());
		//return all_reduce_combiner_.Run(module);


		for (int i=0;i<10;i++){


			std::unique_ptr<HloModule> cloned_module = pop_module->Clone(pop_module->config(),"");
			fusion_node_evaluations_.clear();

			bool changed = false;
			module_ = cloned_module.get();

			computation_list_ = (module_->MakeNonfusionComputationsSorted());//GetComputeLists(module_);
			//randomFusionQueue_map_.clear();
			//do_not_duplicate_map_.clear();
			//reachability_map_.clear();
			//do_not_duplicate = nullptr;

			if (std::getenv("PRINT"))std::cout<<"  in parameter init"<<std::endl;

			/*for (auto* computation : computation_list_) {
			computation_ = computation;
			if (std::getenv("PRINT"))std::cout<<"  in HloReachabilityMap::Buildt"<<std::endl;

			reachability_ = HloReachabilityMap::Build(computation_);
			if (std::getenv("PRINT"))std::cout<<"  out HloReachabilityMap::Build"<<std::endl;

			HloInstructionSet local_do_not_duplicate;
			if (std::getenv("PRINT"))std::cout<<"  in ComputeGloballyUnfusible"<<std::endl;

			local_do_not_duplicate = ComputeGloballyUnfusible(computation_->MakeInstructionPostOrder());
			if (std::getenv("PRINT"))std::cout<<"  out ComputeGloballyUnfusible"<<std::endl;

			do_not_duplicate_map_.emplace(computation_,local_do_not_duplicate);
			reachability_map_.emplace(computation_,reachability_);

			}*/
			if (std::getenv("PRINT"))std::cout<<"  out parameter init"<<std::endl;

			auto start = system_clock::now();

			for(int i = 0; i<5; i++){
				TF_ASSIGN_OR_RETURN(bool changed_this_time, RandomFuseOnce());
				changed = changed ||changed_this_time;
			}
			auto end = system_clock::now();
			auto duration = duration_cast<microseconds>(end - start);
			double realtime = double(duration.count()) * microseconds::period::num / microseconds::period::den;

			//std::cout<<"FuseOnce time:"<<realtime/5<<std::endl;



			if(changed){

				auto est_start = system_clock::now();
				try{
					auto estimate = GetEstimation(module_);

				}catch(exception& e){
					std::cout<<"Exception happen:"<<std::endl;
					cloned_module.reset();
					module_ = nullptr;
					continue;
				}
				auto est_end = system_clock::now();
				auto est_duration = duration_cast<microseconds>(est_end - est_start);
				double est_realtime = double(est_duration.count()) * microseconds::period::num / microseconds::period::den;

				//std::cout<<"Estimation time:"<<est_realtime<<std::endl;


				std::cout<<"  estimation:"<<estimate<<std::endl;

				if (estimate<best_estimation_){
					best_estimation_ = estimate;
					std::cout<<"better estimation:"<<estimate<<std::endl;
					best_module_ =module_->Clone(module_->config(),"");
				}
				if((best_estimation_>0&&estimate<1.1*best_estimation_)||(best_estimation_<0&&1.1*estimate<best_estimation_)){
					std::cout<<"  Push current  module in priority queue"<<std::endl;
					sampled_modules_.emplace(estimate,std::move(cloned_module));

				}else{

				  cloned_module.reset();
				}
				module_ = nullptr;

				/*else if (sampled_modules_.size()<10){
					sampled_modules_.emplace(estimate,cloned_module);
				}*/
				if(sampled_modules_.size()>50){
				  sampled_modules_.erase(std::prev( sampled_modules_.end() ));
				}

		  }




		}
		unique_pop_module.reset();

	}


	std::cout<<"The best estimation find:"<<best_estimation_<<std::endl;

	return true;


}




}  // namespace gpu
}  // namespace xla

