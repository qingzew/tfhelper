# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import re
import os
import tensorflow as tf

from tflearn.utils import check_dir_name, check_restore_tensor

# Fix for TF 0.12
try:
    SummaryWriter = tf.summary.FileWriter
    MergeSummary = tf.summary.merge
except Exception:
    SummaryWriter = tf.train.SummaryWriter
    MergeSummary = tf.merge_summary

class Trainer(object):
    """Trainer
    Arguments:
        tensorboard_dir: `str`. Tensorboard log directory.
            Default: "/tmp/tf_logs/".
        checkpoint_path: `str`. Path to store model checkpoints. If None,
            no model checkpoint will be saved. Default: None.
        max_checkpoints: `int` or None. Maximum amount of checkpoints. If
            None, no limit. Default: None.
        keep_checkpoint_every_n_hours: `float`. Number of hours between each
            model checkpoints.
        random_seed: `int`. Random seed, for test reproductivity.
            Default: None.
    """

    def __init__(self, tensorboard_dir = "/tmp/tf_logs/",
                 checkpoint_path = None, max_checkpoints = None,
                 keep_checkpoint_every_n_hours = 10000.0, random_seed = None,
                 max_steps = 10000):

        if random_seed:
            tf.set_random_seed(random_seed)

        gs = tf.GPUOptions(per_process_gpu_memory_fraction = 0)
        config = tf.ConfigProto(log_device_placement = False,
                            inter_op_parallelism_threads = 0,
                            intra_op_parallelism_threads = 0,
                            gpu_options = gs,
                            allow_soft_placement = True)
        self.session = tf.Session(config = config)

        self.tensorboard_dir = check_dir_name(tensorboard_dir)
        self.checkpoint_path = checkpoint_path
        self.max_steps = max_steps
        self.max_checkpoints = max_checkpoints
        self.keep_checkpoint_every_n_hours= keep_checkpoint_every_n_hours

        self.func_map = {}

    def set_saver(self):
        # Saver for saving a model
        self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints,
                                keep_checkpoint_every_n_hours = self.keep_checkpoint_every_n_hours)

        # Saver for restoring a model (With exclude variable list)
        all_vars = tf.get_collection(tf.GraphKeys.VARIABLES)
        excl_vars = tf.get_collection(tf.GraphKeys.EXCL_RESTORE_VARS)
        to_restore = [item for item in all_vars if check_restore_tensor(item, excl_vars)]
        self.restorer = tf.train.Saver(var_list = to_restore,
                                    max_to_keep = self.max_checkpoints,
                                    keep_checkpoint_every_n_hours = self.keep_checkpoint_every_n_hours)

        # A second Saver, that only restore trainable variables
        to_restore_trainvars = [item for item in tf.trainable_variables()
                                    if check_restore_tensor(item, excl_vars)]
        self.restorer_trainvars = tf.train.Saver(var_list = to_restore_trainvars,
                                            max_to_keep = self.max_checkpoints,
                                            keep_checkpoint_every_n_hours = self.keep_checkpoint_every_n_hours)

    def save(self, model_file, global_step = None):
        """ save.

        Save a Tensorflow model

        Arguments:
            model_file: `str`. Saving path of tensorflow model
            global_step: `int`. The training step to append to the
                model file name (optional).

        """
        # Temp workaround for tensorflow 0.7.0 dict proto serialization issue
        try:
            # Try latest api
            l = tf.get_collection_ref("summary_tags")
            l4 = tf.get_collection_ref(tf.GraphKeys.GRAPH_CONFIG)
        except Exception:
            l = tf.get_collection("summary_tags")
            l4 = tf.get_collection(tf.GraphKeys.GRAPH_CONFIG)
        l_stags = list(l)
        l4_stags = list(l4)
        del l[:]
        del l4[:]

        try:
            # Try latest api
            l1 = tf.get_collection_ref(tf.GraphKeys.DATA_PREP)
            l2 = tf.get_collection_ref(tf.GraphKeys.DATA_AUG)
        except Exception:
            l1 = tf.get_collection(tf.GraphKeys.DATA_PREP)
            l2 = tf.get_collection(tf.GraphKeys.DATA_AUG)
        l1_dtags = list(l1)
        l2_dtags = list(l2)
        del l1[:]
        del l2[:]

        try: # Do not save exclude variables
            l3 = tf.get_collection_ref(tf.GraphKeys.EXCL_RESTORE_VARS)
        except Exception:
            l3 = tf.get_collection(tf.GraphKeys.EXCL_RESTORE_VARS)
        l3_tags = list(l3)
        del l3[:]

        self.saver.save(self.session, model_file, global_step = global_step)

        # 0.7 workaround, restore values
        for t in l_stags:
            tf.add_to_collection("summary_tags", t)
        for t in l4_stags:
            tf.add_to_collection(tf.GraphKeys.GRAPH_CONFIG, t)
        for t in l1_dtags:
            tf.add_to_collection(tf.GraphKeys.DATA_PREP, t)
        for t in l2_dtags:
            tf.add_to_collection(tf.GraphKeys.DATA_AUG, t)
        for t in l3_tags:
            tf.add_to_collection(tf.GraphKeys.EXCL_RESTORE_VARS, t)

    def restore(self, model_file, trainable_variable_only = False, variable_name_map = None, scope_for_restore = None,
                create_new_session = True, verbose = False):
        """ restore.

        Restore a Tensorflow model

        Arguments:
            model_file: path of tensorflow model to restore
            trainable_variable_only: If True, only restore trainable variables.
            variable_name_map: - a (pattern, repl) tuple providing a regular expression pattern
                                 and replacement, which is applied to variable names, before
                                 restoration from the model file
                               - OR, a function map_func, used to perform the mapping, called as:
                                 name_in_file = map_func(existing_var_op_name)
                                 The function may return None to indicate a variable is not to be
                                 restored.
            scope_for_restore: string specifying the scope to limit to, when restoring variables.
                               Also removes the scope name prefix from the var name to use when restoring.
            create_new_session: Set to False if the current session is to be kept.
                                Set to True (the default) to create a new session, and re-init all variables.
            verbose           : Set to True to see a printout of what variables are being restored,
                                when using scope_for_restore or variable_name_map

        """
        # TF 0.12 Fix
        if not os.path.isabs(model_file):
            model_file = os.path.abspath(os.path.join(os.getcwd(), model_file))

        if scope_for_restore is not None:   # allow variables to be restored into a different scope
            sname = scope_for_restore
            def vn_map_func(existing_name):     # variable name map function which removes the scope name, e.g.
                if not existing_name.startswith(sname):  # so that "scope_name/var_name/... is retrieved from var_name/...
                    return None                 # and variables outside of scope_name are not restored
                name_in_file = re.sub("^%s/" % sname, "", existing_name)
                if verbose:
                    print ("[%s] Restoring %s <- %s" % (sname, existing_name, name_in_file))
                return name_in_file
            variable_name_map = vn_map_func

        if variable_name_map is not None:   # general-purpose remapping of variable names (name in file vs existing name)
            if type(variable_name_map) == tuple:  # tuple interpreted as regular expression pattern substitution
                (pattern, repl) = variable_name_map
                def vn_map_func(existing_name):
                    name_in_file = re.sub(pattern, repl, existing_name)
                    if verbose:
                        print ("Restoring %s <- %s" % (existing_name, name_in_file))
                    return name_in_file
            else:
                vn_map_func = variable_name_map     # allow arbitrary user-provided mapping function
            if trainable_variable_only:     # restore either trainingable variables only, or all variables
                to_restore = self.to_restore_trainvars
            else:
                to_restore = self.to_restore
            renamed_to_restore = {vn_map_func(v.op.name): v for v in to_restore}
            if None in renamed_to_restore:
                renamed_to_restore.pop(None)
            restorer = tf.train.Saver(var_list = renamed_to_restore)
            restorer.restore(self.session, model_file)
        elif not trainable_variable_only:
            self.restorer.restore(self.session, model_file)
        else:
            self.restorer_trainvars.restore(self.session, model_file)

    def train(self):
        pass

    def test(self):
        pass

    def register(self, name):
        def func_wrapper(func):
            self.func_map[name] = func
            return func
        return func_wrapper

    def call_method(self, name = None):
        func = self.func_map.get(name, None)
        if func is None:
            raise Exception("No function registered against - " + str(name))
        return func()


if __name__ == '__main__':
    pass
