""" AdaBelief for TensorFlow 1.x."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import resource_variable_ops


class AdaBeliefOptimizer(tf.train.Optimizer):
    """
    It implements the AdaBeliefOptimizer proposed by
    Juntang Zhuang et al. in [AdaBelief Optimizer: Adapting stepsizes by the belief
    in observed gradients](https://arxiv.org/abs/2010.07468).
    Contributor(s):
        Jerry Yu [cryu854] <cryu854@gmail.com>

    Inherits from: tf.train.Optimizer.
    Example of usage:
    ```python
    from adabelief_tf import AdaBeliefOptimizer
    opt = AdaBeliefOptimizer(learning_rate=1e-3, epsilon=1e-14, rectify=False)
    ```
    """

    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-14,
        weight_decay=0.0,
        rectify=True,
        amsgrad=False,
        sma_threshold=5.0,
        total_steps=0,
        warmup_proportion=0.1,
        min_lr=0.0,
        name="AdaBeliefOptimizer",
        use_locking=False,
        print_change_log = True,
        **kwargs):
        r"""Construct a new AdaBelief optimizer.
        Args:
            learning_rate: A `Tensor` or a floating point value, or a schedule
                that is a `tf.keras.optimizers.schedules.LearningRateSchedule`.
                The learning rate.
            beta_1: A float value or a constant float tensor.
                The exponential decay rate for the 1st moment estimates.
            beta_2: A float value or a constant float tensor.
                The exponential decay rate for the 2nd moment estimates.
            epsilon: A small constant for numerical stability.
            weight_decay: A `Tensor` or a floating point value, or a schedule
                that is a `tf.keras.optimizers.schedules.LearningRateSchedule`.
                Weight decay for each parameter.
            rectify: boolean. Whether to enable rectification as in RectifiedAdam
            amsgrad: boolean. Whether to apply AMSGrad variant of this
                algorithm from the paper "On the Convergence of Adam and
                beyond".
            sma_threshold. A float value.
                The threshold for simple mean average.
            total_steps: An integer. Total number of training steps.
                Enable warmup by setting a positive value.
            warmup_proportion: A floating point value.
                The proportion of increasing steps.
            min_lr: A floating point value. Minimum learning rate after warmup.
            name: Optional name for the operations created when applying
                gradients. Defaults to "AdaBeliefOptimizer".
            **kwargs: keyword arguments. Allowed to be {`lr`, `decay`}. `decay` 
                is included for backward compatibility to allow time inverse
                decay of learning rate. `lr` is included for backward
                compatibility, recommended to use `learning_rate` instead.
        """
        super(AdaBeliefOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta_1
        self._beta2 = beta_2
        self._epsilon = epsilon
        self.amsgrad = amsgrad
        self.rectify = rectify

        decay = kwargs.pop("decay", 0.0)
        if decay < 0.:
            raise ValueError("decay cannot be less than 0: {}".format(decay))
        self._initial_decay = decay
        self._decay = self._initial_decay
        self._weight_decay = weight_decay
        self._sma_threshold = sma_threshold
        self._total_steps = int(total_steps)
        self._warmup_proportion = warmup_proportion
        self._min_lr = min_lr
        self._has_weight_decay = weight_decay != 0.0
        self._initial_total_steps = total_steps

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None
        self._decay_t = None
        self._weight_decay_t = None
        self._total_steps_t = None
        self._warmup_proportion_t = None
        self._min_lr_t = None
        self._sma_threshold_t = None

    def _get_beta_accumulators(self):
        with ops.init_scope():
            if context.executing_eagerly():
                graph = None
            else:
                graph = ops.get_default_graph()
            return (self._get_non_slot_variable("beta1_power", graph=graph),
                    self._get_non_slot_variable("beta2_power", graph=graph))
          
    def _get_step(self):
        with ops.init_scope():
            if context.executing_eagerly():
                graph = None
            else:
                graph = ops.get_default_graph()
            return self._get_non_slot_variable("step", graph=graph)

    def _create_slots(self, var_list):
        # Create the step, beta1 and beta2 accumulators on the same device as the first variable.
        first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(
            initial_value=self._beta1, name="beta1_power", colocate_with=first_var)
        self._create_non_slot_variable(
            initial_value=self._beta2, name="beta2_power", colocate_with=first_var)
        self._create_non_slot_variable(
            initial_value=1, name="step", colocate_with=first_var)

        # Create slots.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)
            if self.amsgrad:
                self._zeros_slot(v, "vhat", self._name)

    def _prepare(self):
        lr = self._call_if_callable(self._lr)
        beta1 = self._call_if_callable(self._beta1)
        beta2 = self._call_if_callable(self._beta2)
        epsilon = self._call_if_callable(self._epsilon)
        decay = self._call_if_callable(self._decay)
        weight_decay = self._call_if_callable(self._weight_decay)
        total_steps = self._call_if_callable(self._total_steps)
        warmup_proportion = self._call_if_callable(self._warmup_proportion)
        min_lr = self._call_if_callable(self._min_lr)
        sma_threshold = self._call_if_callable(self._sma_threshold)

        self._lr_t = ops.convert_to_tensor(lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(epsilon, name="epsilon")
        self._decay_t = ops.convert_to_tensor(decay, name="decay")
        self._weight_decay_t = ops.convert_to_tensor(weight_decay, name="weight_decay")
        self._total_steps_t = ops.convert_to_tensor(total_steps, name="total_steps")
        self._warmup_proportion_t = ops.convert_to_tensor(warmup_proportion, name="warmup_proportion")
        self._min_lr_t = ops.convert_to_tensor(min_lr, name="min_lr")
        self._sma_threshold_t = ops.convert_to_tensor(sma_threshold, name="sma_threshold")

    def _decayed_lr(self, var_dtype):
        """Get decayed learning rate as a Tensor with dtype=var_dtype."""
        lr_t = tf.cast(self._lr_t, var_dtype)
        if self._initial_decay > 0.:
            local_step = tf.cast(self._get_step(), var_dtype)
            decay_t = tf.cast(self._decay_t, var_dtype)
            lr_t = lr_t / (1. + decay_t * local_step)
        return lr_t

    def _apply_dense(self, grad, var):
        return self._resource_apply_dense(grad, var)

    def _resource_apply_dense(self, grad, var):
        beta_1_power, beta_2_power = self._get_beta_accumulators()
        beta_1_power = tf.cast(beta_1_power, var.dtype.base_dtype)
        beta_2_power = tf.cast(beta_2_power, var.dtype.base_dtype)
        local_step = tf.cast(self._get_step(), var.dtype.base_dtype)
        lr_t = self._decayed_lr(var.dtype.base_dtype)
        wd_t = tf.cast(self._weight_decay_t, var.dtype.base_dtype)
        beta_1_t = tf.cast(self._beta1_t, var.dtype.base_dtype)
        beta_2_t = tf.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = tf.cast(self._epsilon_t, var.dtype.base_dtype)

        # warmup
        if self._initial_total_steps > 0:
            total_steps = tf.cast(self._total_steps_t, var.dtype.base_dtype)
            warmup_steps = total_steps * tf.cast(self._warmup_proportion_t, var.dtype.base_dtype)
            min_lr = tf.cast(self._min_lr_t, var.dtype.base_dtype)
            decay_steps = tf.maximum(total_steps - warmup_steps, 1)
            decay_rate = (min_lr - lr_t) / decay_steps
            lr_t = tf.cond(
                local_step <= warmup_steps,
                lambda: lr_t * (local_step / warmup_steps),
                lambda: lr_t + decay_rate * tf.minimum(local_step - warmup_steps, decay_steps)
            )

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_t = tf.assign(
            m, beta_1_t * m + (1.0 - beta_1_t) * grad, use_locking=self._use_locking
        )
        m_corr_t = m_t / (1.0 - beta_1_power)

        # v_t = beta2 * v + (1 - beta2) * (g_t - m_t) * (g_t - m_t)
        v = self.get_slot(var, "v")
        v_t = tf.assign(
            v, beta_2_t * v + (1.0 - beta_2_t) * tf.square(grad - m_t) + epsilon_t,
            use_locking=self._use_locking,
        )

        # amsgrad
        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat_t = tf.assign(vhat, tf.maximum(vhat, v_t), use_locking=self._use_locking)
            v_corr_t = tf.sqrt(vhat_t / (1.0 - beta_2_power))
        else:
            vhat_t = None
            v_corr_t = tf.sqrt(v_t / (1.0 - beta_2_power))


        # rectify
        if self.rectify:
            sma_inf = 2.0 / (1.0 - beta_2_t) - 1.0
            sma_t = sma_inf - 2.0 * local_step * beta_2_power / (1.0 - beta_2_power)
            r_t = tf.sqrt(
                (sma_t - 4.0)
                / (sma_inf - 4.0)
                * (sma_t - 2.0)
                / (sma_inf - 2.0)
                * sma_inf
                / sma_t
            )
            sma_threshold = tf.cast(self._sma_threshold_t, var.dtype.base_dtype)
            var_t = tf.cond(
                sma_t >= sma_threshold,
                lambda: r_t * m_corr_t / (v_corr_t + epsilon_t),
                lambda: m_corr_t
            )
        else:
            var_t = m_corr_t / (v_corr_t + epsilon_t)

        # weight_decay
        if self._has_weight_decay:
            var_t += wd_t * var

        # update
        var_update = tf.assign_sub(var, lr_t * var_t, use_locking=self._use_locking)
 
        updates = [var_update, m_t, v_t]
        if self.amsgrad:
            updates.append(vhat_t)
        return tf.group(*updates)

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        beta_1_power, beta_2_power = self._get_beta_accumulators()
        beta_1_power = tf.cast(beta_1_power, var.dtype.base_dtype)
        beta_2_power = tf.cast(beta_2_power, var.dtype.base_dtype)
        local_step = tf.cast(self._get_step(), var.dtype.base_dtype)
        lr_t = self._decayed_lr(var.dtype.base_dtype)
        wd_t = tf.cast(self._weight_decay_t, var.dtype.base_dtype)
        beta_1_t = tf.cast(self._beta1_t, var.dtype.base_dtype)
        beta_2_t = tf.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = tf.cast(self._epsilon_t, var.dtype.base_dtype)

        # warmup
        if self._initial_total_steps > 0:
            total_steps = tf.cast(self._total_steps_t, var.dtype.base_dtype)
            warmup_steps = total_steps * tf.cast(self._warmup_proportion_t, var.dtype.base_dtype)
            min_lr = tf.cast(self._min_lr_t, var.dtype.base_dtype)
            decay_steps = tf.maximum(total_steps - warmup_steps, 1)
            decay_rate = (min_lr - lr_t) / decay_steps
            lr_t = tf.cond(
                local_step <= warmup_steps,
                lambda: lr_t * (local_step / warmup_steps),
                lambda: lr_t + decay_rate * tf.minimum(local_step - warmup_steps, decay_steps)
            )

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = (1.0 - beta_1_t) * grad
        m_t = tf.assign(m, beta_1_t * m, use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = scatter_add(m, indices, m_scaled_g_values)
        m_corr_t = m_t / (1.0 - beta_1_power)

        # v_t = beta2 * v + (1 - beta2) * (g_t - m_t) * (g_t - m_t)
        v = self.get_slot(var, "v")
        m_t_indices = tf.gather(m_t, indices)
        v_scaled_g_values = (1.0 - beta_2_t) * tf.square(grad - m_t_indices)
        v_t = tf.assign(v, v * beta_2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = scatter_add(v, indices, v_scaled_g_values + epsilon_t)

        # amsgrad
        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat_t = tf.assign(vhat, tf.maximum(vhat, v_t), use_locking=self._use_locking)
            v_corr_t = tf.sqrt(vhat_t / (1.0 - beta_2_power))
        else:
            vhat_t = None
            v_corr_t = tf.sqrt(v_t / (1.0 - beta_2_power))

        # rectify
        if self.rectify:
            sma_inf = 2.0 / (1.0 - beta_2_t) - 1.0
            sma_t = sma_inf - 2.0 * local_step * beta_2_power / (1.0 - beta_2_power)
            r_t = tf.sqrt(
                (sma_t - 4.0)
                / (sma_inf - 4.0)
                * (sma_t - 2.0)
                / (sma_inf - 2.0)
                * sma_inf
                / sma_t
            )
            sma_threshold = tf.cast(self._sma_threshold_t, var.dtype.base_dtype)
            var_t = tf.cond(
                sma_t >= sma_threshold,
                lambda: r_t * m_corr_t / (v_corr_t + epsilon_t),
                lambda: m_corr_t
            )
        else:
            var_t = m_corr_t / (v_corr_t + epsilon_t)

        # weight_decay
        if self._has_weight_decay:
            var_t += wd_t * var

        # update
        var_update = tf.assign_sub(var, lr_t * var_t, use_locking=self._use_locking)
 
        updates = [var_update, m_t, v_t]
        if self.amsgrad:
            updates.append(vhat_t)
        return tf.group(*updates)

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values,
            var,
            grad.indices,
            lambda x, i, v: tf.scatter_add(  # pylint: disable=g-long-lambda
                x,
                i,
                v,
                use_locking=self._use_locking))

    def _resource_scatter_add(self, x, i, v):
        with ops.control_dependencies(
            [resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
            return x.value()

    def _resource_apply_sparse(self, grad, var, indices):
        return self._apply_sparse_shared(grad, var, indices,
                                         self._resource_scatter_add)

    def _finish(self, update_ops, name_scope):
        # Update the power accumulators.
        with ops.control_dependencies(update_ops):
            beta1_power, beta2_power = self._get_beta_accumulators()
            step = self._get_step()
            with ops.colocate_with(step):
                update_beta1 = beta1_power.assign(
                                beta1_power * self._beta1_t, use_locking=self._use_locking)
                update_beta2 = beta2_power.assign(
                                beta2_power * self._beta2_t, use_locking=self._use_locking)
                update_step = step.assign(
                                step + 1, use_locking=self._use_locking)
        return tf.group(
            *update_ops + [update_beta1, update_beta2, update_step], name=name_scope)
