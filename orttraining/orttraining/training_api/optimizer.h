// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/inference_session.h"
#include "core/session/environment.h"

#include "orttraining/training_api/module.h"

namespace onnxruntime {
namespace training {
namespace api {

/**
 * @brief States belonging to one specific trainable Parameter.
 *   Momentum states for each Parameter.
 *   For Adam optimizer, it looks like:
 *     { "moment_0": OrtValue, "moment_1": OrtValue,}.
 */
struct ParameterOptimizerState {
  std::unordered_map<std::string, OrtValue> momentum_named_states;
};

/**
 * @brief States belong to one specific group of trainable Parameters.
 */
struct GroupOptimizerState {
  int64_t step = 0;
  float initial_lr = 0.001f;  // Default value used in torch AdamW

  // Adaptive learning rate as training proceeds. Be noted, learning_rate can be
  // restored by lr scheduler from given step and initial_lr, though, we still save/load this in checkpoint.
  float learning_rate{initial_lr};
  std::unordered_map<std::string, ParameterOptimizerState> param_named_optimizer_states;
};

/**
 * @brief States belong to all groups of trainable Parameters.
 * Besides, also maintain a pointer of DataTransferManager* that is owned by InferenceSession.
 * This is used to do Tensor copy in the file saving stage.
 */
struct OptimizerCheckpointState {
 public:
  std::unordered_map<std::string, std::shared_ptr<GroupOptimizerState>> group_named_optimizer_states;
  const DataTransferManager* optimizer_session_data_transfer_mgr;
};

struct OptimizerAlgorithmBase {
  OptimizerAlgorithmBase(const std::vector<std::string>& momentum_keys,
                         const std::vector<std::string>& optimizer_states_inputs)
      : momentum_keys(momentum_keys), optimizer_states_inputs(optimizer_states_inputs) {}
  std::vector<std::string> momentum_keys;
  std::vector<std::string> optimizer_states_inputs;
};

struct AdamWOptimizerAlgorithm : public OptimizerAlgorithmBase {
  AdamWOptimizerAlgorithm() : OptimizerAlgorithmBase({"momentum0", "momentum1"},
                                                     {"first_order_moments", "second_order_moments"}) {}
};

struct SGDOptimizerV2Algorithm : public OptimizerAlgorithmBase {
  SGDOptimizerV2Algorithm() : OptimizerAlgorithmBase({"momentum0"},
                                                     {"first_order_moments"}) {}
};

struct OptimizerAlorithmFactory {
  static std::shared_ptr<OptimizerAlgorithmBase> CreateInstance(const std::string& optim_path_or_bytes,
                                                                int32_t& group_count);
};

struct Optimizer {
  friend struct LRSchedulerBase;

 public:
  // Initialize an optimizer module from an ORT inference session with loaded
  // training ONNX model For each parameter, initialize the OptimizerState based
  // on the graph input's ValueInfoProto if the parameter doesn't have it already.
  Optimizer(const std::string& optim_path_or_bytes,
            const std::unordered_map<std::string, std::shared_ptr<Parameter>>& named_parameters,
            const onnxruntime::SessionOptions& session_options,
            const Environment& env,
            const std::vector<std::shared_ptr<IExecutionProvider>>& providers);

  Status Step();

  /**
   * @brief Get current optimizer state and store the CPU copy in optimizer_checkpoint_states.
   *
   * Be noted there are a copy underlying between devices (CPUtoCPU, CUDAtoCPU).
   * @return Status
   */
  Status GetStateDict(OptimizerCheckpointState& optimizer_checkpoint_states);

  /**
   * @brief Load states from optimizer_checkpoint_states into current optimizer state.
   *
   * Be noted there are a copy underlying between devices (CPUtoCPU, CPUtoCUDA).
   * @return Status
   */
  Status LoadStateDict(const OptimizerCheckpointState& optimizer_checkpoint_states);

  Status SetLearningRate(float lr) {
    optimizer_state_.learning_rate = lr;
    return Status::OK();
  }

  float GetLearningRate() const noexcept {
    return optimizer_state_.learning_rate;
  }

  Status SetInitialLearningRate(float initial_lr) {
    optimizer_state_.initial_lr = initial_lr;
    optimizer_state_.learning_rate = initial_lr;
    return Status::OK();
  }

 private:
  int64_t GetStep() const {
    return optimizer_state_.step;
  }

  // Generates optimizer momentum states for parameters that require grad.
  Status GenerateMomentumNamedStates();
  // Constructs the ortvalue inputs to be fed to the graph
  // at each step
  Status ConstructInputs();

  /**
 * @brief Copy optimizer states between src and dest across different devices.
 *
 * Be noted: upon calling GetStateDict/LoadStateDict, a copy will be done
 * (potentially across two same/different devices).
 *
 * @param src_group_optimizer_state Can be optimizer state (on CPU/CUDA) passed in by caller;
 *  or the state currently maintained (on CPU/CUDA).
 * @param dst_group_optimizer_state Can be optimizer state (on CPU/CUDA) passed in by caller;
 *  or the state currently maintained (on CPU/CUDA).
 * @param src_strict_match If true, for any trainable param, its states MUST exist in src_group_optimizer_state;
 *  otherwise fail. If False, the param state will be skipped to copy.
 * @param dst_strict_match If true, for any trainable param, its states MUST exist in dst_group_optimizer_state;
 *  otherwise fail. If False and src state exist,  a new state will be created in dst_group_optimizer_state.
 * @return Status
 */
  Status CopyOptimizerState(const GroupOptimizerState& src_group_optimizer_state,
                            GroupOptimizerState& dst_group_optimizer_state,
                            bool src_strict_match = false,
                            bool dst_strict_match = false);

  std::shared_ptr<OptimizerAlgorithmBase> optimizer_algo_shared_ptr_;
  std::unique_ptr<onnxruntime::InferenceSession> optim_sess_;
  const std::unordered_map<std::string, std::shared_ptr<Parameter>> named_parameters_;
  GroupOptimizerState optimizer_state_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<OrtValue> inputs_;

  int32_t group_count_{0};
};

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
