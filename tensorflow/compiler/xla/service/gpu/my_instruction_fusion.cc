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
#include <mpi.h>
#include <assert.h>

#include <curl/curl.h>

#include <iostream>
#include <experimental/random>

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
std::string GetEstimation(HloModule* module_) {
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
	return readBuffer;
}


// A FusionQueue that uses reverse post order.
//
// We want to be able to remove arbitrary instructions from the post order and
// also compare positions of instructions in the post order. To make this
// possible, create vector of instructions in post order and create a map from
// HloInstruction* to the instruction's index in the vector. An instruction is
// "removed" from the vector by setting it's element to nullptr.
class RandomFusionQueue : public FusionQueue {
 public:
  explicit RandomFusionQueue(HloComputation* computation) {
    post_order_ = computation->MakeInstructionPostOrder();

    for (size_t i = 0; i < post_order_.size(); ++i) {
      InsertOrDie(&post_order_index_, post_order_[i], i);
    }
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_);
	const char* produce_sample=std::getenv("PRODUCE_SAMPLE");
	if (produce_sample){
		produce_sample_=true;
	}else{
		produce_sample_=false;

	}


  }

  std::pair<HloInstruction*, std::vector<int64>>
  DequeueLastInstructionAndOperandsToFuseInOrder(){
	    // Instructions are "removed" from the post order by nulling out the element
	    // in the vector, so if the pointer is null, continue to the next
	    // instruction in the sort.
		if (post_order_.empty()) {
		  return std::pair<HloInstruction*, std::vector<int64>>{nullptr, {}};
		}

		int random_number=post_order_.size()-1;
	    while (!post_order_.empty() && post_order_[random_number] == nullptr) {
	      post_order_.erase(post_order_.begin()+random_number);
	    }
	    if (post_order_.empty()) {
	      return std::pair<HloInstruction*, std::vector<int64>>{nullptr, {}};
	    }
	    // We want to iterate in reverse post order, so remove from the back of the
	    // vector.
	    HloInstruction* instruction = post_order_[random_number];
	    post_order_.erase(post_order_.begin()+random_number);

	    CHECK(instruction != nullptr);
	    // Remove instruction from the index map to ensure the vector and map stay
	    // consistent.
	    post_order_index_.erase(instruction);

	    // Consider each operand of this instruction for fusion into this
	    // instruction. We want to consider the operands in a particular order to
	    // avoid creating duplicate instruction clones in the fusion instruction.
	    // For example, consider the following expression:
	    //
	    //   A = ...
	    //   B = op(A)
	    //   C = op(A, B)
	    //
	    // If we are considering the operands of C for fusion into C. We might
	    // fuse A or B first. If we fuse A first, we get:
	    //
	    //   A = ...
	    //   B = op(A)
	    //   C_fusion = { A' = ...
	    //                C' = op(A', B) }
	    //
	    // Where A' and C' are clones of A and C, respectively. Now only B is an
	    // operand of the fusion instruction C_fusion, so then we fuse B:
	    //
	    //   A = ...
	    //   B = op(A)
	    //   C_fusion = { A' = ...
	    //                B' = op(A)
	    //                C' = op(A', B') }
	    //
	    // Now A is an operand of C_fusion again, so we then fuse A (again!):
	    //
	    //   A = ...
	    //   B = op(A)
	    //   C_fusion = { A' = ...
	    //                A" = ..
	    //                B' = op(A")
	    //                C' = op(A', B') }
	    //
	    // We prevent this duplication by considering the operands in the order
	    // they appear int the queue. In the example, this ensures that B will be
	    // considered before A.
	    //
	    // We store the original indices of the operands to pass to ShouldFuse.
	    std::vector<int64> sorted_operand_numbers;
	    sorted_operand_numbers.reserve(instruction->operands().size());
	    for (int i = 0; i < instruction->operands().size(); ++i) {
	      // This will happen if we have two possible instructions to fuse the
	      // same operand into; once the operand is fused into one instruction,
	      // the other instruction will get a new get-tuple-element as its
	      // operand, which is not in the queue.
	      // TODO(tjoerg): Look into fusing past these multi-output fuse points.
	      if (!ContainsKey(post_order_index_, instruction->mutable_operand(i))) {
	        continue;
	      }
	      sorted_operand_numbers.push_back(i);
	    }
	    absl::c_sort(sorted_operand_numbers, [&](int64 i, int64 j) {
	      // Instructions with higher priority in the queue come first.
	      return (FindOrDie(post_order_index_, instruction->mutable_operand(i)) >
	              FindOrDie(post_order_index_, instruction->mutable_operand(j)));
	    });
	    return std::make_pair(instruction, sorted_operand_numbers);


  }

  std::pair<HloInstruction*, std::vector<int64>>
  DequeueNextInstructionAndOperandsToFuseInOrder() override {
    // Instructions are "removed" from the post order by nulling out the element
    // in the vector, so if the pointer is null, continue to the next
    // instruction in the sort.
	if (post_order_.empty()) {
	  return std::pair<HloInstruction*, std::vector<int64>>{nullptr, {}};
	}

	int random_number=0;
	if(!produce_sample_){
		random_number= std::experimental::randint(0, int(post_order_.size())-1);

	}else{
		if (proc_==0){
			random_number= std::experimental::randint(0, int(post_order_.size())-1);
			//std::cout<<"random_number: "<<random_number<<std::endl;
			MPI_Bcast(&random_number, sizeof(random_number), MPI_BYTE, 0, MPI_COMM_WORLD);

		}else{
			MPI_Bcast(&random_number, sizeof(random_number), MPI_BYTE, 0, MPI_COMM_WORLD);

		}
	}

    while (!post_order_.empty() && post_order_[random_number] == nullptr) {
      post_order_.erase(post_order_.begin()+random_number);
    }
    if (post_order_.empty()) {
      return std::pair<HloInstruction*, std::vector<int64>>{nullptr, {}};
    }
    // We want to iterate in reverse post order, so remove from the back of the
    // vector.
    HloInstruction* instruction = post_order_[random_number];
    post_order_.erase(post_order_.begin()+random_number);

    CHECK(instruction != nullptr);
    // Remove instruction from the index map to ensure the vector and map stay
    // consistent.
    post_order_index_.erase(instruction);

    // Consider each operand of this instruction for fusion into this
    // instruction. We want to consider the operands in a particular order to
    // avoid creating duplicate instruction clones in the fusion instruction.
    // For example, consider the following expression:
    //
    //   A = ...
    //   B = op(A)
    //   C = op(A, B)
    //
    // If we are considering the operands of C for fusion into C. We might
    // fuse A or B first. If we fuse A first, we get:
    //
    //   A = ...
    //   B = op(A)
    //   C_fusion = { A' = ...
    //                C' = op(A', B) }
    //
    // Where A' and C' are clones of A and C, respectively. Now only B is an
    // operand of the fusion instruction C_fusion, so then we fuse B:
    //
    //   A = ...
    //   B = op(A)
    //   C_fusion = { A' = ...
    //                B' = op(A)
    //                C' = op(A', B') }
    //
    // Now A is an operand of C_fusion again, so we then fuse A (again!):
    //
    //   A = ...
    //   B = op(A)
    //   C_fusion = { A' = ...
    //                A" = ..
    //                B' = op(A")
    //                C' = op(A', B') }
    //
    // We prevent this duplication by considering the operands in the order
    // they appear int the queue. In the example, this ensures that B will be
    // considered before A.
    //
    // We store the original indices of the operands to pass to ShouldFuse.
    std::vector<int64> sorted_operand_numbers;
    sorted_operand_numbers.reserve(instruction->operands().size());
    for (int i = 0; i < instruction->operands().size(); ++i) {
      // This will happen if we have two possible instructions to fuse the
      // same operand into; once the operand is fused into one instruction,
      // the other instruction will get a new get-tuple-element as its
      // operand, which is not in the queue.
      // TODO(tjoerg): Look into fusing past these multi-output fuse points.
      if (!ContainsKey(post_order_index_, instruction->mutable_operand(i))) {
        continue;
      }
      sorted_operand_numbers.push_back(i);
    }
    absl::c_sort(sorted_operand_numbers, [&](int64 i, int64 j) {
      // Instructions with higher priority in the queue come first.
      return (FindOrDie(post_order_index_, instruction->mutable_operand(i)) >
              FindOrDie(post_order_index_, instruction->mutable_operand(j)));
    });
    return std::make_pair(instruction, sorted_operand_numbers);
  }

  std::vector<int64> GetSortedOperandNumbers(HloInstruction*instruction){
	    std::vector<int64> sorted_operand_numbers;
	    sorted_operand_numbers.reserve(instruction->operands().size());
	    for (int i = 0; i < instruction->operands().size(); ++i) {
	      // This will happen if we have two possible instructions to fuse the
	      // same operand into; once the operand is fused into one instruction,
	      // the other instruction will get a new get-tuple-element as its
	      // operand, which is not in the queue.
	      // TODO(tjoerg): Look into fusing past these multi-output fuse points.
	      if (!ContainsKey(post_order_index_, instruction->mutable_operand(i))) {
	        continue;
	      }
	      sorted_operand_numbers.push_back(i);
	    }
	    absl::c_sort(sorted_operand_numbers, [&](int64 i, int64 j) {
	      // Instructions with higher priority in the queue come first.
	      return (FindOrDie(post_order_index_, instruction->mutable_operand(i)) >
	              FindOrDie(post_order_index_, instruction->mutable_operand(j)));
	    });


	    return sorted_operand_numbers;

  }


  void OnFusingInstruction(HloInstruction* fusion,
                           HloInstruction* original_producer,
                           HloInstruction* original_consumer) override {
    // Fusing an instruction into a fusion instruction can change the operand
    // set of the fusion instruction. For simplicity just re-enqueue the
    // instruction and reconsider it for further fusion in the next iteration.
    InsertOrDie(&post_order_index_, fusion, post_order_.size());
    post_order_.push_back(fusion);
  }

  void RemoveInstruction(HloInstruction* instruction) override {
    post_order_[FindOrDie(post_order_index_, instruction)] = nullptr;
    post_order_index_.erase(instruction);
  }

  const std::vector<bool>* FusionConfiguration() override {
    return &fusion_config_;
  }

 private:
  std::vector<HloInstruction*> post_order_;
  absl::flat_hash_map<HloInstruction*, int> post_order_index_;

  int proc_;
  bool produce_sample_;
  std::vector<bool> fusion_config_;

};





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



std::shared_ptr<FusionQueue> MyGpuInstructionFusion::GetRandomFusionQueue(
    HloComputation* computation){


	auto res = randomFusionQueue_map_.find(computation);
	if (res==randomFusionQueue_map_.end()){

		auto emplace_res = randomFusionQueue_map_.emplace(computation,std::make_shared<RandomFusionQueue>(computation));
		/*if(!emplace_res.second){
			std::cout<<"emplace fail, already exist"<<std::endl;
		}else{
			std::cout<<"emplace success"<<std::endl;

		}*/
		res = emplace_res.first;
	}

	return res->second;

	//return std::make_shared<RandomFusionQueue>(computation);


}

bool MyGpuInstructionFusion::ShouldFuseInexpensiveChecks(HloInstruction* consumer,
                                                       int64 operand_index) {
  HloInstruction* producer = consumer->mutable_operand(operand_index);

  // Output fusions are not currently supported on GPUs.
  if (producer->opcode() == HloOpcode::kFusion) {
    VLOG(4) << "Producer " << producer->name() << " is a fusion op";
    //std::cout << "Producer " << producer->name() << " is a fusion op"<<std::endl;

    return false;
  }
  // Cost condition: not fuse (simple, expensive producers) and (consumers who
  // reuse operand elements).
  if (producer->opcode() != HloOpcode::kFusion && is_expensive(*producer) &&
      ReusesOperandElements(consumer, operand_index)) {
    VLOG(4) << "Do not fuse simple, expensive producer " << producer->name()
            << " and consumer which reuses operand elements.";
    //std::cout << "Do not fuse simple, expensive producer " << producer->name()<< " and consumer which reuses operand elements."<<std::endl;
    return false;
  }

  if (!IsProducerConsumerFusible(*producer, *consumer) ||
      !InstructionFusion::ShouldFuse(consumer, operand_index)) {
    VLOG(4) << "Producer " << producer->name()
            << " is not fusible or should not be fused.";
    //std::cout << "Producer " << producer->name()<< " is not fusible or should not be fused."<<std::endl;
    return false;
  }
  return true;
}

bool MyGpuInstructionFusion::ShouldFuse(HloInstruction* consumer,
                                      int64 operand_index) {
  if (!ShouldFuseInexpensiveChecks(consumer, operand_index)) {
    VLOG(5) << "Not fusing inexpensive checks of operand " << operand_index
            << " of " << consumer->ToString();
    //std::cout << "Not fusing inexpensive checks of operand " << operand_index<< " of " << consumer->ToString()<<std::endl;
    return false;
  }
  auto producer = consumer->operand(operand_index);

  // The following checks are potentially expensive.
  if (FusionWouldBeTooLarge(*consumer, *producer,
                            /*is_consumer_producer_fusion=*/true)) {
    VLOG(5) << "Fusion of (" << producer->ToString() << ") into ("
            << consumer->ToString() << ") would be too large";
    //std::cout << "Fusion of (" << producer->ToString() << ") into ("<< consumer->ToString() << ") would be too large"<<std::endl;
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
    //std::cout << "Fusion of " << producer->name() << " into " << consumer->name()<< " would result in overly large code duplication."<<std::endl;
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


StatusOr<HloInstruction*> MyGpuInstructionFusion::FuseSpecificInstruction(HloInstruction* instruction){
    if (instruction == nullptr) {
      return nullptr;
    }
    if (!instruction->IsFusible() &&
        instruction->opcode() != HloOpcode::kFusion) {
        std::cout << "instruction (" << instruction->name() << ") is not fusible"<<std::endl;
      return nullptr;
    }

    auto fusion_queue =GetRandomFusionQueue(computation_);
    HloInstruction* fusion_instruction = nullptr;

    auto sorted_operand_numbers =static_cast<RandomFusionQueue*>(fusion_queue.get())->GetSortedOperandNumbers(instruction);

    std::cout<<"sorted_operand_numbers length:"<<sorted_operand_numbers.size()<<std::endl;

    for (int64 i : sorted_operand_numbers) {
      HloInstruction* operand = instruction->mutable_operand(i);
      VLOG(5) << "Considering fusion of: " << instruction->ToString()
              << " with operand " << operand->name();

      if (!operand->IsFusible()) {
      	std::cout << "Operand (" << operand->name() << ") is not fusible"<<std::endl;
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
      	//std::cout << "do_not_duplicate has " << operand->name()<<std::endl;
      //}
      //if (!ShouldFuse(instruction, i)){
      	//std::cout << "The " <<i<<"operand:"<<operand->name()<<"cannot fused to "<<"instruction: "<<instruction->name()<<std::endl;
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
    return fusion_instruction;

}


StatusOr<bool> MyGpuInstructionFusion::RandomFuseOnce(){
	bool changed = false;
	int fuse_count = 0;
	int random_number= std::experimental::randint(0, int(computation_list_.size())-1);
	computation_ = computation_list_[random_number];
    reachability_ = reachability_map_.find(computation_)->second;
    // If we allow duplications, we need to compute which instructions we do not
    // want to duplicate based on a global analysis of the graph.

    do_not_duplicate =do_not_duplicate_map_.find(computation_)->second;
    auto fusion_queue = GetRandomFusionQueue(computation_);

    while (!changed){

    	auto next_entry =
    	  fusion_queue->DequeueNextInstructionAndOperandsToFuseInOrder();

    	HloInstruction* instruction = next_entry.first;
    	if (instruction == nullptr) {

    		computation_list_.erase(computation_list_.begin()+random_number);
    		std::cout<<"fusion_queue nullptr"<<std::endl;
    		break;
    	}

    	int random_times= std::experimental::randint(1,10);
    	for (int i=0;i<random_times;i++){

    		TF_ASSIGN_OR_RETURN(instruction, FuseSpecificInstruction(instruction));
    		if (instruction == nullptr) {
    			std::cout<<"FuseSpecificInstruction nullptr"<<std::endl;
    			break;
    		}else{
    			//dequeue just fused instruction.
    			auto fused_entry =static_cast<RandomFusionQueue*>(fusion_queue.get())->DequeueLastInstructionAndOperandsToFuseInOrder();
    			auto fused_instruction = fused_entry.first;
    			assert(fused_instruction==instruction);
    			changed = true;
    		    ++fuse_count;

    		}

    	}

    }


	std::cout << "Fusion count: " << fuse_count<<std::endl;
	return changed;

}



StatusOr<bool> MyGpuInstructionFusion::Run(HloModule* module){
  fusion_node_evaluations_.clear();
  //TF_RETURN_IF_ERROR(InstructionFusion::Run(module).status());
  //TF_RETURN_IF_ERROR(multi_output_fuser_.Run(module).status());
  //return all_reduce_combiner_.Run(module);

  bool changed = false;
  module_ = module;
  int64 fuse_count = 0;

  computation_list_ = module->MakeNonfusionComputationsSorted();


  for (auto* computation : module->MakeNonfusionComputationsSorted()) {
	computation_ = computation;
	reachability_ = HloReachabilityMap::Build(computation_);
	HloInstructionSet do_not_duplicate;
	do_not_duplicate = ComputeGloballyUnfusible(computation->MakeInstructionPostOrder());
	do_not_duplicate_map_.emplace(computation,do_not_duplicate);
	reachability_map_.emplace(computation,reachability_);

  }
  for (int i=0;i<500;i++){


	  TF_ASSIGN_OR_RETURN(changed, RandomFuseOnce());
	  if(changed){

		  auto estimate = GetEstimation(module_);

		  std::cout<<"estimation:"<<estimate<<std::endl;

	  }




  }



  return changed;


}












}  // namespace gpu
}  // namespace xla

