// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/execution_provider.h"
#include "core/framework/TensorSeq.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/inference_session.h"
#include "core/session/environment.h"

#include "orttraining/training_api/utils.h"
#include "orttraining/training_api/optimizer.h"

namespace onnxruntime {
namespace training {
namespace api {

namespace {

constexpr char GROUP_ZERO_NAME[] = "group0";
static const std::string FullQualifiedName_AdamWOptimizer(std::string(kMSDomain) + ":AdamWOptimizer");
static const std::string FullQualifiedName_SGDOptimizerV2(std::string(kMSDomain) + ":SGDOptimizerV2");

static const std::vector<std::string> CommonOptimizerInputs{"learning_rate", "step", "params", "gradients"};

Status GraphInputsAreExpected(gsl::span<std::string> actual_graph_inputs,
                              gsl::span<std::string> expected_graph_inputs) {
  const auto stringify = [](const auto& container) {
    if (container.empty()) {
      return std::string("[]");
    }
    std::string container_str("[");
    for (const auto& val : container) {
      container_str += std::string(val) + ", ";
    }
    container_str.pop_back();
    container_str.back() = ']';

    return container_str;
  };

  const auto construct_unexpected_input_status = [&stringify](const auto& actual_inputs, const auto& expected_inputs) {
    std::ostringstream error_stream;
    error_stream << "Invalid graph inputs."
                 << "\n\tExpected: " << stringify(expected_inputs)
                 << "\n\tActual: " << stringify(actual_inputs);
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, error_stream.str());
  };

  if (actual_graph_inputs.size() != expected_graph_inputs.size()) {
    return construct_unexpected_input_status(actual_graph_inputs, expected_graph_inputs);
  }

  for (size_t input_idx = 0; input_idx < expected_graph_inputs.size(); ++input_idx) {
    if (actual_graph_inputs[input_idx] != expected_graph_inputs[input_idx]) {
      return construct_unexpected_input_status(actual_graph_inputs, expected_graph_inputs);
    }
  }

  return Status::OK();
}

}  // namespace

std::shared_ptr<OptimizerAlgorithmBase> OptimizerAlorithmFactory::CreateInstance(
    const std::string& optim_path_or_bytes, int32_t& group_count) {
  std::shared_ptr<Model> model;
  ORT_ENFORCE(Model::Load(optim_path_or_bytes, model, nullptr, logging::LoggingManager::DefaultLogger()).IsOK());
  Graph& graph = model->MainGraph();
  std::unordered_map<std::string, int32_t> opt_type_to_freq_map;
  for (auto& node : graph.Nodes()) {
    if (node.Domain() == kMSDomain && (node.OpType() == "AdamWOptimizer" || node.OpType() == "SGDOptimizerV2")) {
      auto opt_op_type = node.Domain() + ":" + node.OpType();
      if (opt_type_to_freq_map.find(opt_op_type) == opt_type_to_freq_map.end()) {
        opt_type_to_freq_map[opt_op_type] = 0;
      }

      opt_type_to_freq_map[opt_op_type] += 1;
    }
  }

  ORT_ENFORCE(opt_type_to_freq_map.size() == 1U, "Only support one type of optimizer algorithm, but got: " +
                                                     std::to_string(opt_type_to_freq_map.size()));
  auto opt_it = opt_type_to_freq_map.begin();
  group_count = opt_it->second;

  // TODO: to support multiple group, need create a mapping between each group to its parameter list.
  if (opt_it->first == FullQualifiedName_AdamWOptimizer) {
    return std::make_shared<AdamWOptimizerAlgorithm>();
  } else if (opt_it->first == FullQualifiedName_SGDOptimizerV2) {
    return std::make_shared<SGDOptimizerV2Algorithm>();
  } else {
    ORT_NOT_IMPLEMENTED("Not implemented for optimizer algo: " + opt_it->first);
  }
}

Status Optimizer::GenerateMomentumNamedStates() {
  auto& param_named_optimizer_states = optimizer_state_.param_named_optimizer_states;
  auto& optim_sess_state = optim_sess_->GetSessionState();
  for (auto& pair : named_parameters_) {
    if (pair.second->RequiresGrad()) {
      param_named_optimizer_states.insert({pair.first, ParameterOptimizerState()});
      ParameterOptimizerState& cur_param_optimizer_states = param_named_optimizer_states[pair.first];
      for (auto& state_name : optimizer_algo_shared_ptr_->momentum_keys) {
        OrtValue param_state;
        ORT_ENFORCE(utils::OrtValueZeroLike(optim_sess_state, pair.second->Data(), param_state).IsOK(),
                    "Error generating moment state for ", pair.first);
        cur_param_optimizer_states.momentum_named_states.insert({state_name, std::move(param_state)});
      }
    }
  }

  return Status::OK();
}

// Constructs the ortvalue inputs to be fed to the graph at each step
Status Optimizer::ConstructInputs() {
  auto& param_named_optimizer_states = optimizer_state_.param_named_optimizer_states;

  std::vector<Tensor> params, grads;
  std::vector<std::vector<Tensor>> list_of_momentums;
  list_of_momentums.resize(optimizer_algo_shared_ptr_->momentum_keys.size());

  // Collect all the non user defined inputs from the named_parameters_.
  for (auto& [parameter_name, parameter] : named_parameters_) {
    if (parameter->RequiresGrad()) {
      // Collect parameters and prepare for tensorseq creation
      auto* param_tensor = parameter->Data().GetMutable<Tensor>();
      params.emplace_back(
          Tensor(param_tensor->DataType(), param_tensor->Shape(),
                 param_tensor->MutableDataRaw(), param_tensor->Location()));

      // Collect gradients and prepare for tensorseq creation
      auto* grad_tensor = parameter->Gradient().GetMutable<Tensor>();
      grads.emplace_back(
          Tensor(grad_tensor->DataType(), grad_tensor->Shape(),
                 grad_tensor->MutableDataRaw(), grad_tensor->Location()));

      // Collect moments and prepare for tensorseq creation
      for (size_t m_index = 0; m_index < optimizer_algo_shared_ptr_->momentum_keys.size(); ++m_index) {
        auto* moment_tensor =
            param_named_optimizer_states.at(parameter_name)
                .momentum_named_states.at(optimizer_algo_shared_ptr_->momentum_keys[m_index])
                .GetMutable<Tensor>();
        list_of_momentums[m_index].emplace_back(
            Tensor(moment_tensor->DataType(), moment_tensor->Shape(),
                   moment_tensor->MutableDataRaw(), moment_tensor->Location()));
      }
    }
  }

  const auto tensorseq_inserter = [](auto& tensors, auto* inputs) {
    ORT_ENFORCE(!tensors.empty(), "Tensors vector cannot be empty while building a tensor sequence.");

    auto tensor_seq = std::make_unique<TensorSeq>(tensors.front().DataType());
    tensor_seq->Reserve(tensors.size());
    for (auto& tensor : tensors) {
      tensor_seq->Add(std::move(tensor));
    }
    inputs->emplace_back(
        OrtValue(tensor_seq.release(), DataTypeImpl::GetType<TensorSeq>(),
                 DataTypeImpl::GetType<TensorSeq>()->GetDeleteFunc()));
  };

  // Add the params/grads as tensorseq ortvalues to inputs
  tensorseq_inserter(params, &inputs_);
  tensorseq_inserter(grads, &inputs_);
  // Add all other momentums  as tensorseq ortvalues to inputs.
  for (auto& m : list_of_momentums) {
    tensorseq_inserter(m, &inputs_);
  }

  return Status::OK();
}

Optimizer::Optimizer(const std::string& optim_path_or_bytes,
                     const std::unordered_map<std::string, std::shared_ptr<Parameter>>& named_parameters,
                     const onnxruntime::SessionOptions& session_options,
                     const Environment& env,
                     const std::vector<std::shared_ptr<IExecutionProvider>>& providers)
    : optim_sess_(std::make_unique<InferenceSession>(session_options, env)), named_parameters_(named_parameters) {
  for (const auto& execution_provider : providers) {
    ORT_THROW_IF_ERROR(optim_sess_->RegisterExecutionProvider(execution_provider));
  }

  ORT_THROW_IF_ERROR(optim_sess_->Load(optim_path_or_bytes));
  ORT_THROW_IF_ERROR(optim_sess_->Initialize());

  utils::GetGraphInputOutputNames(optim_sess_, input_names_, output_names_);

  optimizer_algo_shared_ptr_ = OptimizerAlorithmFactory::CreateInstance(optim_path_or_bytes, group_count_);
  ORT_ENFORCE(group_count_ == 1, "Group count can only be 1, but got: " + std::to_string(group_count_));
  ORT_ENFORCE(optimizer_algo_shared_ptr_, "optimizer_algo_shared_ptr_ should not be nullptr.");

  InlinedVector<std::string> all_input_names;
  all_input_names.reserve(CommonOptimizerInputs.size() + optimizer_algo_shared_ptr_->optimizer_states_inputs.size());
  all_input_names.insert(all_input_names.end(), CommonOptimizerInputs.begin(),
                         CommonOptimizerInputs.end());
  all_input_names.insert(all_input_names.end(), optimizer_algo_shared_ptr_->optimizer_states_inputs.begin(),
                         optimizer_algo_shared_ptr_->optimizer_states_inputs.end());
  ORT_THROW_IF_ERROR(GraphInputsAreExpected(input_names_, all_input_names));

  // TODO: add multiple group support.
  ORT_THROW_IF_ERROR(GenerateMomentumNamedStates());

  ORT_THROW_IF_ERROR(ConstructInputs());
}

Status Optimizer::Step() {
  OrtValue learning_rate_input, step_input;
  utils::WrapInOrtValue<float>(optimizer_state_.learning_rate, &learning_rate_input);
  // Use step count + 1 before running optimizer step.
  // This is necessary since bias correction uses the step
  // as a power. Using power of 0 is wrong.
  utils::WrapInOrtValue<int64_t>(optimizer_state_.step + 1, &step_input);
  std::vector<OrtValue> feeds({learning_rate_input, step_input});
  feeds.insert(feeds.end(), inputs_.begin(), inputs_.end());

  std::vector<OrtValue> outputs;
  auto status = optim_sess_->Run(RunOptions(), input_names_, feeds, output_names_, &outputs);
  ORT_THROW_IF_ERROR(status);

  // Extract step output and update
  if (utils::GetScalarFromOrtValue<int64_t>(outputs[0]) == 1LL) {
    optimizer_state_.step++;
  }

  return Status::OK();
}

Status Optimizer::CopyOptimizerState(const GroupOptimizerState& src_group_optimizer_state,
                                     GroupOptimizerState& dst_group_optimizer_state,
                                     bool src_strict_match,
                                     bool dst_strict_match) {
  ORT_RETURN_IF_NOT(optim_sess_, "optimizer session not initialized");
  const DataTransferManager& sess_data_transfer_manager = optim_sess_->GetDataTransferManager();
  auto& optim_sess_state = optim_sess_->GetSessionState();
  auto& src_param_named_optimizer_states = src_group_optimizer_state.param_named_optimizer_states;
  auto& dst_param_named_optimizer_states = dst_group_optimizer_state.param_named_optimizer_states;

  if (src_strict_match && dst_strict_match)
    ORT_ENFORCE(src_param_named_optimizer_states.size() == dst_param_named_optimizer_states.size());

  for (auto& pair : named_parameters_) {
    if (pair.second->RequiresGrad()) {
      bool src_exist = src_param_named_optimizer_states.find(pair.first) != src_param_named_optimizer_states.cend();
      bool dst_exist = dst_param_named_optimizer_states.find(pair.first) != dst_param_named_optimizer_states.cend();
      if (src_strict_match) {
        ORT_ENFORCE(src_exist, "Parameter ", pair.first, " not found in the source optimizer checkpoint states.");
      }

      if (dst_strict_match) {
        ORT_ENFORCE(dst_exist, "Parameter ", pair.first, " not found in the dest optimizer checkpoint states.");
      }

      if (!src_exist) {
        // If nothing can be copied, then we just skip.
        continue;
      }

      const std::unordered_map<std::string, OrtValue>& src_momentum_named_states =
          src_param_named_optimizer_states.at(pair.first).momentum_named_states;

      // If dst_exist is false, we will create a new item in dst_momentum_named_states, and continue the copy.
      std::unordered_map<std::string, OrtValue>& dst_momentum_named_states =
          dst_param_named_optimizer_states[pair.first].momentum_named_states;

      if (dst_exist)
        ORT_ENFORCE(src_momentum_named_states.size() == dst_momentum_named_states.size());

      for (auto& src_momentum_state_pair : src_momentum_named_states) {
        if (!dst_exist) {
          ORT_ENFORCE(utils::OrtValueZeroLike(optim_sess_state, src_momentum_state_pair.second,
                                              dst_momentum_named_states[src_momentum_state_pair.first])
                          .IsOK(),
                      "Error generating moment state for ", src_momentum_state_pair.first);
        }
        const Tensor& src_momentum_state_tensor = src_momentum_state_pair.second.Get<Tensor>();
        OrtValue& dst_momentum_state = dst_momentum_named_states[src_momentum_state_pair.first];
        auto* dst_momentum_state_tensor = dst_momentum_state.GetMutable<Tensor>();

        ORT_THROW_IF_ERROR(sess_data_transfer_manager.CopyTensor(src_momentum_state_tensor,
                                                                 *dst_momentum_state_tensor));
      }
    }
  }

  dst_group_optimizer_state.initial_lr = src_group_optimizer_state.initial_lr;
  dst_group_optimizer_state.step = src_group_optimizer_state.step;
  dst_group_optimizer_state.learning_rate = src_group_optimizer_state.learning_rate;

  return Status::OK();
}

Status Optimizer::GetStateDict(OptimizerCheckpointState& optimizer_checkpoint_state) {
  auto& grouped_optimizer_states = optimizer_checkpoint_state.group_named_optimizer_states;
  ORT_ENFORCE(grouped_optimizer_states.size() == 0, "Passed in optimizer_checkpoint_state must be empty.");

  auto output_group_optimizer_state = std::make_shared<GroupOptimizerState>();
  ORT_THROW_IF_ERROR(CopyOptimizerState(optimizer_state_, *output_group_optimizer_state,
                                        true /*src_strict_match*/, false /*dst_strict_match*/));

  // To support multiple groups, Optimizer constructor need accept informations for groupping.
  grouped_optimizer_states.insert({GROUP_ZERO_NAME, output_group_optimizer_state});

  // TODO(pengwa): remove passing sess_data_transfer_manager outside, since we do the copy here already.
  const DataTransferManager& sess_data_transfer_manager = optim_sess_->GetDataTransferManager();
  optimizer_checkpoint_state.optimizer_session_data_transfer_mgr = &sess_data_transfer_manager;
  return Status::OK();
}

Status Optimizer::LoadStateDict(const OptimizerCheckpointState& optimizer_checkpoint_states) {
  auto group_optimizer_state_it =
      optimizer_checkpoint_states.group_named_optimizer_states.find(GROUP_ZERO_NAME);

  ORT_ENFORCE(group_optimizer_state_it != optimizer_checkpoint_states.group_named_optimizer_states.cend(),
              "Group 0 not found in the optimizer checkpoint states.");
  ORT_THROW_IF_ERROR(CopyOptimizerState(*group_optimizer_state_it->second, optimizer_state_,
                                        true /*src_strict_match*/, true /*dst_strict_match*/));

  return Status::OK();
}

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
