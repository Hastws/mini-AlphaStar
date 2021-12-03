#!/usr/bin/env python
# -*- coding: utf-8 -*-

"Library for RL algorithms."

# modified from AlphaStar pseudo-code
import traceback
import collections
import itertools

import numpy as np

import torch
import torch.nn as nn

from alphastarmini.core.rl.rl_utils import Trajectory
from alphastarmini.core.rl.action import ArgsActionLogits

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP

__author__ = "Ruo-Ze Liu"

debug = False


def reverse_seq(sequence, sequence_lengths=None):
    """Reverse sequence along dim 0.
    Args:
      sequence: Tensor of shape [T, B, ...].
      sequence_lengths: (optional) tensor of shape [B]. If `None`, only reverse
        along dim 0.
    Returns:
      Tensor of same shape as sequence with dim 0 reversed up to sequence_lengths.
    """
    if sequence_lengths is None:
        return torch.flip(sequence, [0])
    else:
        raise NotImplementedError


def lambda_returns(values_tp1, rewards, discounts, lambdas=0.8):
    """Computes lambda returns.

    Refer to the following for a similar function:
    https://github.com/deepmind/trfl/blob/2c07ac22512a16715cc759f0072be43a5d12ae45/trfl/value_ops.py#L74
    """

    # we only implment the lambda return version in AlphaStar when lambdas=0.8
    # assert lambdas != 1

    # assert v_tp1 = torch.concat([values[1:, :], torch.unsqueeze(bootstrap_value, 0)], axis=0)
    # `back_prop=False` prevents gradients flowing into values and
    # bootstrap_value, which is what you want when using the bootstrapped
    # lambda-returns in an update as targets for values.
    return multistep_forward_view(
        rewards,
        discounts,
        values_tp1,
        lambdas,
        back_prop=False,
        name="0.8_lambda_returns")


def multistep_forward_view(rewards, pcontinues, state_values, lambda_,
                           back_prop=True, sequence_lengths=None,
                           name="multistep_forward_view_op"):
    """Evaluates complex backups (forward view of eligibility traces).
      ```python
      result[t] = rewards[t] +
          pcontinues[t]*(lambda_[t]*result[t+1] + (1-lambda_[t])*state_values[t])
      result[last] = rewards[last] + pcontinues[last]*state_values[last]
      ```
      This operation evaluates multistep returns where lambda_ parameter controls
      mixing between full returns and boostrapping.
      ```
    Args:
      rewards: Tensor of shape `[T, B]` containing rewards.
      pcontinues: Tensor of shape `[T, B]` containing discounts.
      state_values: Tensor of shape `[T, B]` containing state values.
      lambda_: Mixing parameter lambda.
          The parameter can either be a scalar or a Tensor of shape `[T, B]`
          if mixing is a function of state.
      back_prop: Whether to backpropagate.
      sequence_lengths: Tensor of shape `[B]` containing sequence lengths to be
        (reversed and then) summed, same as in `scan_discounted_sum`.
      name: Sets the name_scope for this op.
    Returns:
        Tensor of shape `[T, B]` containing multistep returns.
    """

    # Regroup:
    #   result[t] = (rewards[t] + pcontinues[t]*(1-lambda_)*state_values[t]) +
    #               pcontinues[t]*lambda_*result[t + 1]
    # Define:
    #   sequence[t] = rewards[t] + pcontinues[t]*(1-lambda_)*state_values[t]
    #   discount[t] = pcontinues[t]*lambda_
    # Substitute:
    #   result[t] = sequence[t] + discount[t]*result[t + 1]
    # Boundary condition:
    #   result[last] = rewards[last] + pcontinues[last]*state_values[last]
    # Add and subtract the same quantity at BC:
    #   state_values[last] =
    #       lambda_*state_values[last] + (1-lambda_)*state_values[last]
    # This makes:
    #   result[last] =
    #       (rewards[last] + pcontinues[last]*(1-lambda_)*state_values[last]) +
    #       pcontinues[last]*lambda_*state_values[last]
    # Substitute in definitions for sequence and discount:
    #   result[last] = sequence[last] + discount[last]*state_values[last]
    # Define:
    #   initial_value=state_values[last]
    # We get the following recurrent relationship:
    #   result[last] = sequence[last] + decay[last]*initial_value
    #   result[k] = sequence[k] + decay[k] * result[k + 1]
    # This matches the form of scan_discounted_sum:
    #   result = scan_sum_with_discount(sequence, discount,
    #                                   initial_value = state_values[last])  
    sequence = rewards + pcontinues * state_values * (1 - lambda_)
    print("sequence", sequence) if debug else None
    print("sequence.shape", sequence.shape) if debug else None

    discount = pcontinues * lambda_
    print("discount", discount) if debug else None
    print("discount.shape", discount.shape) if debug else None

    return scan_discounted_sum(sequence, discount, state_values[-1],
                               reverse=True, sequence_lengths=sequence_lengths,
                               back_prop=back_prop)


def scan_discounted_sum(sequence, decay, initial_value, reverse=False,
                        sequence_lengths=None, back_prop=True,
                        name="scan_discounted_sum"):
    """Evaluates a cumulative discounted sum along dimension 0.
      ```python
      if reverse = False:
        result[1] = sequence[1] + decay[1] * initial_value
        result[k] = sequence[k] + decay[k] * result[k - 1]
      if reverse = True:
        result[last] = sequence[last] + decay[last] * initial_value
        result[k] = sequence[k] + decay[k] * result[k + 1]
      ```
    Respective dimensions T, B and ... have to be the same for all input tensors.
    T: temporal dimension of the sequence; B: batch dimension of the sequence.
      if sequence_lengths is set then x1 and x2 below are equivalent:
      ```python
      x1 = zero_pad_to_length(
        scan_discounted_sum(
            sequence[:length], decays[:length], **kwargs), length=T)
      x2 = scan_discounted_sum(sequence, decays,
                               sequence_lengths=[length], **kwargs)
      ```
    Args:
      sequence: Tensor of shape `[T, B, ...]` containing values to be summed.
      decay: Tensor of shape `[T, B, ...]` containing decays/discounts.
      initial_value: Tensor of shape `[B, ...]` containing initial value.
      reverse: Whether to process the sum in a reverse order.
      sequence_lengths: Tensor of shape `[B]` containing sequence lengths to be
        (reversed and then) summed.
      back_prop: Whether to backpropagate.
      name: Sets the name_scope for this op.
    Returns:
      Cumulative sum with discount. Same shape and type as `sequence`.
    """
    # Note this can be implemented in terms of cumprod and cumsum,
    # approximately as (ignoring boundary issues and initial_value):
    #
    # cumsum(decay_prods * sequence) / decay_prods
    # where decay_prods = reverse_cumprod(decay)
    #
    # One reason this hasn't been done is that multiplying then dividing again by
    # products of decays isn't ideal numerically, in particular if any of the
    # decays are zero it results in NaNs.
    if sequence_lengths is not None:
        raise NotImplementedError

    elems = [sequence, decay]
    if reverse:
        elems = [reverse_seq(s, sequence_lengths) for s in elems]

    elems = [s.unsqueeze(0) for s in elems]
    elems = torch.cat(elems, dim=0) 
    elems = torch.transpose(elems, 0, 1)
    print("elems", elems) if debug else None
    print("elems.shape", elems.shape) if debug else None

    # we change it to a pytorch version
    def scan(foo, x, initial_value, debug=False):
        res = []
        a_ = initial_value.clone().detach()
        print("a_", a_) if debug else None
        print("a_.shape", a_.shape) if debug else None

        res.append(foo(a_, x[0]).unsqueeze(0))
        print("res", res) if debug else None
        print("len(x)", len(x)) if debug else None

        for i in range(1, len(x)):
            print("i", i) if debug else None
            res.append(foo(a_, x[i]).unsqueeze(0))
            print("res", res) if debug else None

            a_ = foo(a_, x[i])
            print("a_", a_) if debug else None
            print("a_.shape", a_.shape) if debug else None

        return torch.cat(res)

    # summed = tf.scan(lambda a, x: x[0] + x[1] * a,
    #                 elems,
    #                 initializer=tf.convert_to_tensor(initial_value),
    #                 parallel_iterations=1,
    #                 back_prop=back_prop)
    summed = scan(lambda a, x: x[0] + x[1] * a, elems, initial_value=initial_value)
    print("summed", summed) if debug else None
    print("summed.shape", summed.shape) if debug else None   

    if reverse:
        summed = reverse_seq(summed, sequence_lengths)

    return summed


def vtrace_advantages(clipped_rhos, rewards, discounts, values, bootstrap_value):
    """Computes v-trace return advantages.

    Refer to the following for a similar function:
    https://github.com/deepmind/trfl/blob/40884d4bb39f99e4a642acdbe26113914ad0acec/trfl/vtrace_ops.py#L154
    see below function "vtrace_from_importance_weights"
    """
    return vtrace_from_importance_weights(rhos=clipped_rhos, discounts=discounts,
                                          rewards=rewards, values=values,
                                          bootstrap_value=bootstrap_value)


VTraceReturns = collections.namedtuple('VTraceReturns', 'vs pg_advantages')


def vtrace_from_importance_weights(
        rhos, discounts, rewards, values, bootstrap_value,
        clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0,
        name='vtrace_from_importance_weights'):
    r"""
    https://github.com/deepmind/trfl/blob/40884d4bb39f99e4a642acdbe26113914ad0acec/trfl/vtrace_ops.py#L154
    V-trace from log importance weights.
    Calculates V-trace actor critic targets as described in
    "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures"
    by Espeholt, Soyer, Munos et al.
    In the notation used throughout documentation and comments, T refers to the
    time dimension ranging from 0 to T-1. B refers to the batch size and
    NUM_ACTIONS refers to the number of actions. This code also supports the
    case where all tensors have the same number of additional dimensions, e.g.,
    `rewards` is `[T, B, C]`, `values` is `[T, B, C]`, `bootstrap_value`
    is `[B, C]`.
    Args:
      log_rhos: A float32 tensor of shape `[T, B, NUM_ACTIONS]` representing the
        log importance sampling weights, i.e.
        log(target_policy(a) / behaviour_policy(a)). V-trace performs operations
        on rhos in log-space for numerical stability.
        # note: in mAS we change it to rhos instead of log_rhos
      discounts: A float32 tensor of shape `[T, B]` with discounts encountered
        when following the behaviour policy.
      rewards: A float32 tensor of shape `[T, B]` containing rewards generated by
        following the behaviour policy.
      values: A float32 tensor of shape `[T, B]` with the value function estimates
        wrt. the target policy.
      bootstrap_value: A float32 of shape `[B]` with the value function estimate
        at time T.
      clip_rho_threshold: A scalar float32 tensor with the clipping threshold for
        importance weights (rho) when calculating the baseline targets (vs).
        rho^bar in the paper. If None, no clipping is applied.
      clip_pg_rho_threshold: A scalar float32 tensor with the clipping threshold
        on rho_s in \rho_s \delta log \pi(a|x) (r + \gamma v_{s+1} - V(x_s)). If
        None, no clipping is applied.
      name: The name scope that all V-trace operations will be created in.
    Returns:
      A VTraceReturns namedtuple (vs, pg_advantages) where:
        vs: A float32 tensor of shape `[T, B]`. Can be used as target to
          train a baseline (V(x_t) - vs_t)^2.
        pg_advantages: A float32 tensor of shape `[T, B]`. Can be used as the
          advantage in the calculation of policy gradients.
    """

    if clip_rho_threshold is not None:
        clip_rho_threshold = torch.tensor(clip_rho_threshold,
                                          dtype=torch.float32, device=values.device)
    if clip_pg_rho_threshold is not None:
        clip_pg_rho_threshold = torch.tensor(clip_pg_rho_threshold,
                                             dtype=torch.float32, device=values.device)

    # Make sure tensor ranks are consistent.
    if clip_rho_threshold is not None:
        clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
    else:
        clipped_rhos = rhos

    cs = torch.clamp(rhos, max=1.)

    # Append bootstrapped value to get [v1, ..., v_t+1]
    values_t_plus_1 = torch.cat(
        [values[1:], bootstrap_value.unsqueeze(0)], axis=0)

    deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)
    print("deltas:", deltas) if debug else None
    print("deltas.shape:", deltas.shape) if debug else None

    # Note that all sequences are reversed, computation starts from the back.
    '''
    Note this code is wrong, we should use zip to concat
    sequences = (
        torch.flip(discounts, dims=[0]),
        torch.flip(cs, dims=[0]),
        torch.flip(deltas, dims=[0]),
    )
    '''
    flip_discounts = torch.flip(discounts, dims=[0])
    flip_cs = torch.flip(cs, dims=[0])
    flip_deltas = torch.flip(deltas, dims=[0])

    sequences = [item for item in zip(flip_discounts, flip_cs, flip_deltas)]

    # V-trace vs are calculated through a 
    # scan from the back to the beginning
    # of the given trajectory.
    def scanfunc(acc, sequence_item):
        discount_t, c_t, delta_t = sequence_item
        return delta_t + discount_t * c_t * acc
    initial_values = torch.zeros_like(bootstrap_value, device=bootstrap_value.device)

    # our implemented scan function for pytorch
    def scan(foo, x, initial_value):
        res = []
        a_ = initial_value.clone().detach()
        res.append(foo(a_, x[0]).unsqueeze(0))

        for i in range(1, len(x)):
            res.append(foo(a_, x[i]).unsqueeze(0))
            a_ = foo(a_, x[i])
        return torch.cat(res)

    vs_minus_v_xs = scan(foo=scanfunc, x=sequences, initial_value=initial_values)

    '''
    # the original tensorflow code
    vs_minus_v_xs = tf.scan(
        fn=scanfunc,
        elems=sequences,
        initializer=initial_values,
        parallel_iterations=1,
        back_prop=False)
        '''
    # Reverse the results back to original order.
    vs_minus_v_xs = torch.flip(vs_minus_v_xs, dims=[0])

    # Add V(x_s) to get v_s.
    vs = torch.add(vs_minus_v_xs, values)

    # Advantage for policy gradient.
    vs_t_plus_1 = torch.cat([vs[1:], bootstrap_value.unsqueeze(0)], axis=0)

    if clip_pg_rho_threshold is not None:
        clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
    else:
        clipped_pg_rhos = rhos

    pg_advantages = (clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values))

    # Make sure no gradients backpropagated through the returned values.
    return VTraceReturns(vs=vs.detach(), pg_advantages=pg_advantages.detach())


def upgo_returns(values, rewards, discounts, bootstrap):
    """Computes the UPGO return targets.

    Args:
      values: Estimated state values. Shape [T, B].
      rewards: Rewards received moving to the next state. Shape [T, B].
      discounts: If the step is NOT final. Shape [T, B].
      bootstrap: Bootstrap values. Shape [B].
    Returns:
      UPGO return targets. Shape [T, B].
    """
    print("rewards", rewards) if debug else None
    print("discounts", discounts) if debug else None

    # we change it to pytorch version
    # next_values = np.concatenate((values[1:], np.expand_dims(bootstrap, axis=0)), axis=0)
    next_values = torch.cat([values[1:], bootstrap.unsqueeze(0)], dim=0)
    print("next_values", next_values) if debug else None
    print("next_values.shape", next_values.shape) if debug else None

    # Upgo can be viewed as a lambda return! The trace continues (i.e. lambda =
    # 1.0) if r_t + V_tp1 > V_t.
    lambdas = (rewards + discounts * next_values) >= values
    print("lambdas", lambdas) if debug else None
    print("lambdas.shape", lambdas.shape) if debug else None

    # change the bool tensor to float tensor
    lambdas = lambdas.float()

    # Shift lambdas left one slot, such that V_t matches indices with lambda_tp1.
    # lambdas = np.concatenate((lambdas[1:], np.ones_like(lambdas[-1:])), axis=0)
    lambdas = torch.cat([lambdas[1:], torch.ones_like(lambdas[-1:], device=lambdas.device)], dim=0)

    return lambda_returns(next_values, rewards, discounts, lambdas)


def generalized_lambda_returns(rewards,
                               pcontinues,
                               values,
                               bootstrap_value,
                               lambda_=1,
                               name="generalized_lambda_returns"):
    """
    # not used actually in mini-AlphaStar
    code at https://github.com/deepmind/trfl/blob/2c07ac22512a16715cc759f0072be43a5d12ae45/trfl/value_ops.py#L74
    Computes lambda-returns along a batch of (chunks of) trajectories.
    For lambda=1 these will be multistep returns looking ahead from each
    state to the end of the chunk, where bootstrap_value is used. If you pass an
    entire trajectory and zeros for bootstrap_value, this is just the Monte-Carlo
    return / TD(1) target.
    For lambda=0 these are one-step TD(0) targets.
    The sequences in the tensors should be aligned such that an agent in a state
    with value `V` transitions into another state with value `V'`, receiving
    reward `r` and pcontinue `p`. Then `V`, `r` and `p` are all at the same index
    `i` in the corresponding tensors. `V'` is at index `i+1`, or in the
    `bootstrap_value` tensor if `i == T`.
    Subtracting `values` from these lambda-returns will yield estimates of the
    advantage function which can be used for both the policy gradient loss and
    the baseline value function loss in A3C / GAE.
    Args:
      rewards: 2-D Tensor with shape `[T, B]`.
      pcontinues: 2-D Tensor with shape `[T, B]`.
      values: 2-D Tensor containing estimates of the state values for timesteps
        0 to `T-1`. Shape `[T, B]`.
      bootstrap_value: 1-D Tensor containing an estimate of the value of the
        final state at time `T`, used for bootstrapping the target n-step
        returns. Shape `[B]`.
      lambda_: an optional scalar or 2-D Tensor with shape `[T, B]`.
      name: Customises the name_scope for this op.
    Returns:
      2-D Tensor with shape `[T, B]`
    """

    if lambda_ == 1:
                # This is actually equivalent to the branch below, just an optimisation
                # to avoid unnecessary work in this case:
        return scan_discounted_sum(
            rewards,
            pcontinues,
            initial_value=bootstrap_value,
            reverse=True,
            back_prop=False,
            name="multistep_returns")
    else:
        v_tp1 = torch.concat([values[1:, :], torch.unsqueeze(bootstrap_value, 0)], axis=0)
        # `back_prop=False` prevents gradients flowing into values and
        # bootstrap_value, which is what you want when using the bootstrapped
        # lambda-returns in an update as targets for values.
        return multistep_forward_view(
            rewards,
            pcontinues,
            v_tp1,
            lambda_,
            back_prop=False,
            name="generalized_lambda_returns")


def entropy(policy_logits, masks):
    # policy_logits shape: [seq_batch_size, channel_size]
    # masks shape: [seq_batch_size, 1]

    softmax = nn.Softmax(dim=-1)
    logsoftmax = nn.LogSoftmax(dim=-1)

    policy = softmax(policy_logits)
    log_policy = logsoftmax(policy_logits)

    ent = torch.sum(-policy * log_policy * masks, axis=-1)  # Aggregate over actions.
    # Normalize by actions available.
    normalized_entropy = ent / torch.log(torch.tensor(policy_logits.shape[-1], 
                                                      dtype=torch.float32, 
                                                      device=policy_logits.device))

    return normalized_entropy


def kl(student_logits, teacher_logits, mask):
    softmax = nn.Softmax(dim=-1)
    logsoftmax = nn.LogSoftmax(dim=-1)

    s_logprobs = logsoftmax(student_logits)
    print("s_logprobs:", s_logprobs) if debug else None
    print("s_logprobs.shape:", s_logprobs.shape) if debug else None    

    t_logprobs = logsoftmax(teacher_logits)
    print("t_logprobs:", t_logprobs) if debug else None
    print("t_logprobs.shape:", t_logprobs.shape) if debug else None  

    teacher_probs = softmax(teacher_logits)
    print("teacher_probs:", teacher_probs) if debug else None
    print("teacher_probs.shape:", teacher_probs.shape) if debug else None

    kl = teacher_probs * (t_logprobs - s_logprobs) * mask
    print("kl:", kl) if debug else None
    print("kl.shape:", kl.shape) if debug else None

    return kl


def compute_unclipped_logrho(behavior_logits, target_logits, actions):
    """Helper function for compute_importance_weights."""

    target_log_prob = log_prob(actions, target_logits, reduction="none")
    print("target_log_prob:", target_log_prob) if debug else None

    behavior_log_prob = log_prob(actions, behavior_logits, reduction="none")
    print("behavior_log_prob:", behavior_log_prob) if debug else None

    subtract = target_log_prob - behavior_log_prob
    print("subtract:", subtract) if debug else None

    return subtract


def compute_importance_weights(behavior_logits, target_logits, actions):
    """Computes clipped importance weights."""

    logrho = compute_unclipped_logrho(behavior_logits, target_logits, actions)
    print("logrho:", logrho) if debug else None
    print("logrho.shape:", logrho.shape) if debug else None

    rho = torch.exp(logrho)
    print("rho:", rho) if debug else None

    # change to pytorch version
    clip_rho = torch.clamp(rho, max=1.)
    print("clip_rho:", clip_rho) if debug else None

    return clip_rho   


def log_prob(actions, logits, reduction="none"):
    """Returns the log probability of taking an action given the logits."""
    # Equivalent to tf.sparse_softmax_cross_entropy_with_logits.

    # note CrossEntropyLoss is $ - log(e^(x_i) / \sum_j{e^{x_j}}) $
    # such for log_prob is actually -CrossEntropyLoss
    loss = torch.nn.CrossEntropyLoss(reduction=reduction)

    # logits: shape [BATCH_SIZE, CLASS_SIZE]
    # actions: shape [BATCH_SIZE]
    loss_result = loss(logits, torch.squeeze(actions, dim=-1))

    # Original AlphaStar pseudocode is wrong
    # AlphaStar: return loss_result

    # change to right log_prob
    the_log_prob = - loss_result

    return the_log_prob
