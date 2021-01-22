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

#include "tensorflow/compiler/xla/service/all_reduce_combiner.h"


namespace xla {
namespace gpu {

class MyGpuInstructionFusion : public GpuInstructionFusion{//,public GpuMultiOutputFusion,public AllReduceCombiner {
 public:
  explicit MyGpuInstructionFusion(bool may_duplicate)
      : GpuInstructionFusion(may_duplicate),multi_output_fuser_(GpuMultiOutputFusion()),all_reduce_combiner_(AllReduceCombiner(0,0)){}//,GpuMultiOutputFusion(),AllReduceCombiner(0,0) {}

  static bool IsExpensive(const HloInstruction& instruction);

  bool ShouldFuse(HloInstruction* consumer, int64 operand_index) override;

  bool ShouldFuseIntoMultiOutput(HloInstruction* consumer,
                                 int64 operand_index) override;
  std::unique_ptr<FusionQueue> GetRandomFusionQueue(
      HloComputation* computation);

  HloInstruction::FusionKind ChooseKind(
      const HloInstruction* producer, const HloInstruction* consumer) override;

  StatusOr<bool> Run(HloModule* module) override;
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

  std::vector<HloInstruction*> instruction_list_;
  std::vector<HloComputation*> computation_list_;




  GpuMultiOutputFusion multi_output_fuser_;
  AllReduceCombiner all_reduce_combiner_;

};

}  // namespace gpu
}  // namespace xla



#endif /* TENSORFLOW_COMPILER_XLA_SERVICE_GPU_MY_INSTRUCTION_FUSION_H_ */
