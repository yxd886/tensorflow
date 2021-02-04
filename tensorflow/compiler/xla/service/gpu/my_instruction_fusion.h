/*
 * my_instruction_fusion.h
 *
 *  Created on: Jan 21, 2021
 *      Author: admin
 */

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_MY_INSTRUCTION_FUSION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_MY_INSTRUCTION_FUSION_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/service/fusion_node_indexing_evaluation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/gpu/multi_output_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/instruction_fusion.h"
#include "tensorflow/compiler/xla/service/fusion_queue.h"

#include "tensorflow/compiler/xla/service/all_reduce_combiner.h"
#include <functional>
#include <assert.h>

#include <curl/curl.h>

#include <iostream>
#include <mpi.h>

#include <experimental/random>

namespace xla {
namespace gpu {




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

	    while (!post_order_.empty() && post_order_.back() == nullptr) {
	      post_order_.pop_back();
	    }
	    if (post_order_.empty()) {
	      return std::pair<HloInstruction*, std::vector<int64>>{nullptr, {}};
	    }
	    // We want to iterate in reverse post order, so remove from the back of the
	    // vector.
	    HloInstruction* instruction = post_order_.back();
	    post_order_.pop_back();

	    CHECK(instruction != nullptr);
	    // Remove instruction from the index map to ensure the vector and map stay
	    // consistent.
	    post_order_index_.erase(instruction);

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
			//if (std::getenv("PRINT"))std::cout<<"random_number: "<<random_number<<std::endl;
			MPI_Bcast(&random_number, sizeof(random_number), MPI_BYTE, 0, MPI_COMM_WORLD);

		}else{
			MPI_Bcast(&random_number, sizeof(random_number), MPI_BYTE, 0, MPI_COMM_WORLD);

		}
	}

    //while (!post_order_.empty() && post_order_[random_number] == nullptr) {
    //  post_order_.erase(post_order_.begin()+random_number);
    //}
	assert(post_order_[random_number] != nullptr);

    // We want to iterate in reverse post order, so remove from the back of the
    // vector.
    HloInstruction* instruction = post_order_[random_number];
    post_order_.erase(post_order_.begin()+random_number);
    for (int i = random_number;i<post_order_.size();i++){
    	post_order_index_[post_order_[i]]=i;
    }


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
	assert(fusion!=nullptr);
    InsertOrDie(&post_order_index_, fusion, post_order_.size());
    post_order_.push_back(fusion);
  }

  void RemoveInstruction(HloInstruction* instruction) override {

		int index = FindOrDie(post_order_index_, instruction);
		post_order_.erase(post_order_.begin()+index);

		for (int i = index;i<post_order_.size();i++){
			post_order_index_[post_order_[i]]=i;
		}

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






class MyGpuInstructionFusion : public GpuInstructionFusion{//,public GpuMultiOutputFusion,public AllReduceCombiner {
 public:
  explicit MyGpuInstructionFusion(bool may_duplicate)
      : GpuInstructionFusion(may_duplicate),multi_output_fuser_(GpuMultiOutputFusion()),all_reduce_combiner_(AllReduceCombiner(0,0)){

  }//,GpuMultiOutputFusion(),AllReduceCombiner(0,0) {}

  static bool IsExpensive(const HloInstruction& instruction);

  bool ShouldFuse(HloInstruction* consumer, int64 operand_index) override;

  bool ShouldFuseIntoMultiOutput(HloInstruction* consumer,
                                 int64 operand_index) override;
  std::unique_ptr<FusionQueue> GetRandomFusionQueue(
      HloComputation* computation);

  StatusOr<bool> RandomFuseOnce();

  std::vector<HloComputation*>* GetComputeLists(HloModule* module);

  StatusOr<HloInstruction*> FuseSpecificInstruction(HloInstruction* instruction,FusionQueue*fusion_queue);

  HloInstruction::FusionKind ChooseKind(
      const HloInstruction* producer, const HloInstruction* consumer) override;

  StatusOr<bool> Run(HloModule* module) override;

  std::unique_ptr<HloModule> best_module_;

 private:
  // This method is called by ShouldFuse() to do all the computationally
  // inexpensive checks whether we should fuse the operand into 'consumer'.
  bool ShouldFuseInexpensiveChecks(HloInstruction* consumer,
                                   int64 operand_index);

  HloInstruction* FuseInstruction(HloInstruction* fusion_instruction,
                                  HloInstruction* producer) override;

  // Keep track of the number of times each instruction inside a fusion node is
  // indexed with different index vectors.
  absl::flat_hash_map<const HloInstruction*, FusionNodeIndexingEvaluation>
      fusion_node_evaluations_;
  absl::flat_hash_map<HloComputation*, RandomFusionQueue> randomFusionQueue_map_;
  absl::flat_hash_map<HloComputation*, std::shared_ptr<HloReachabilityMap>> reachability_map_;
  absl::flat_hash_map<HloComputation*, HloInstructionSet> do_not_duplicate_map_;


  std::vector<HloInstruction*> instruction_list_;
  std::vector<HloComputation*> computation_list_;

  std::map<HloModule*,std::vector<HloComputation*>> module_computation_list_;

  double best_estimation_;



  struct CmpByDouble {
    bool operator()(const double& left, const double& right) {
        return (abs(left - right) > 1e-7) && (left < right);

    }
  };

  std::map<double,std::unique_ptr<HloModule>,CmpByDouble> sampled_modules_;
  std::unique_ptr<HloInstructionSet> do_not_duplicate;



  GpuMultiOutputFusion multi_output_fuser_;
  AllReduceCombiner all_reduce_combiner_;

};

}  // namespace gpu
}  // namespace xla



#endif /* TENSORFLOW_COMPILER_XLA_SERVICE_GPU_MY_INSTRUCTION_FUSION_H_ */
