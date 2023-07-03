import time
import threading

class pipeline_coordinator:
    def __init__(self, stall_policy):
        assert stall_policy in ['drop', 'debug_keep_all']
        self.stall_policy = stall_policy

        # Registration
        self.stage_registration_dict = {}

        # Threading variables
        self.stage_thread_dict = {}

        # This records data flow *after* a stage
        # e.g., self.stage_dataflow_buffer[1] contains output from stage 1
        self.stage_dataflow_buffer = {}

        self.tic_tac_dict = {}
    
    def register_pipeline(self, stage, func, name, input_list=[], output_list=[]):
        if stage == 1:
            assert input_list == []
        assert stage not in self.stage_registration_dict.keys()
        self.stage_registration_dict[stage] = {}
        self.stage_registration_dict[stage]['func'] = func
        self.stage_registration_dict[stage]['name'] = name
        self.stage_registration_dict[stage]['inputs'] = input_list
        self.stage_registration_dict[stage]['outputs'] = output_list

        self.tic_tac_dict[stage] = {
            'total_time': 0,
            'total_calls': 0,
            'min_time': 999,
            'max_time': 0,
        }
    
    def record_time(self, stage, time_elapsed):
        self.tic_tac_dict[stage]['total_time'] += time_elapsed
        self.tic_tac_dict[stage]['total_calls'] += 1
        self.tic_tac_dict[stage]['min_time'] = min(self.tic_tac_dict[stage]['min_time'], time_elapsed)
        self.tic_tac_dict[stage]['max_time'] = max(self.tic_tac_dict[stage]['max_time'], time_elapsed)
    
    def parse_all_stage(self):
        self.max_stage = max(self.stage_registration_dict.keys())
        for stage_num in range(1, self.max_stage + 1):
            self.stage_dataflow_buffer[stage_num] = {
                'lock': threading.Lock(),
                'data': [],
            }
            if stage_num == 1:
                assert self.stage_registration_dict[stage_num]['inputs'] == []
                def thread_func_factory(k):
                    while True:
                        start_cp = time.time()
                        # Pass through the function in the current stage
                        output_dict = self.stage_registration_dict[k]['func']()
                        if isinstance(output_dict, tuple):
                            assert len(output_dict) == len(self.stage_registration_dict[k]['outputs'])
                            output_dict = {k: v for k, v in zip(self.stage_registration_dict[k]['outputs'], output_dict)}
                        else:
                            assert len(self.stage_registration_dict[k]['outputs']) == 1
                            output_dict = {self.stage_registration_dict[k]['outputs'][0]: output_dict}
                        # Write output to the next stage
                        self.stage_dataflow_buffer[k]['lock'].acquire()
                        self.stage_dataflow_buffer[k]['data'].append(output_dict)
                        self.stage_dataflow_buffer[k]['lock'].release()
                        time.sleep(0.001)
                        end_cp = time.time()
                        self.record_time(k, end_cp - start_cp)
                self.stage_thread_dict[stage_num] = threading.Thread(target=thread_func_factory,
                                                                     args=(stage_num, ))
            else:
                def thread_func_factory(k):
                    while True:
                        # Get output from previous stage
                        self.stage_dataflow_buffer[k - 1]['lock'].acquire()
                        input_buffer = self.stage_dataflow_buffer[k - 1]['data']
                        if len(input_buffer) == 0:
                            self.stage_dataflow_buffer[k - 1]['lock'].release()
                            # TODO(roger): use signal event for better call
                            time.sleep(0.001)
                            continue
                        else:
                            start_cp = time.time()
                            if self.stall_policy == 'drop':
                                # Keep only the latest data
                                input_dict = input_buffer.pop(-1)
                                self.stage_dataflow_buffer[k - 1]['data'] = []
                            elif self.stall_policy == 'debug_keep_all':
                                input_dict = input_buffer.pop(0)
                                self.stage_dataflow_buffer[k - 1]['data'] = input_buffer
                            else:
                                raise NotImplementedError
                            self.stage_dataflow_buffer[k - 1]['lock'].release()
                            # Pass through the function in the current stage
                            func_params_dict = {}
                            for var_name in self.stage_registration_dict[k]['inputs']:
                                func_params_dict[var_name] = input_dict[var_name]
                            output_vars = self.stage_registration_dict[k]['func'](**func_params_dict)
                            if k != self.max_stage:
                                output_dict = {}
                                if isinstance(output_vars, tuple):
                                    assert len(output_vars) == len(self.stage_registration_dict[k]['outputs'])
                                    for var_name in self.stage_registration_dict[k]['outputs']:
                                        output_dict[var_name] = output_vars.pop(0)
                                elif output_vars is None:
                                    assert len(self.stage_registration_dict[k]['outputs']) == 0
                                else:
                                    assert len(self.stage_registration_dict[k]['outputs']) == 1
                                    output_dict[self.stage_registration_dict[k]['outputs'][0]] = output_vars
                                # Copy used input data for synchronization
                                for input_k in input_dict.keys():
                                    assert input_k not in output_dict
                                    output_dict[input_k] = input_dict[input_k]
                                # Write output to the next stage
                                self.stage_dataflow_buffer[k]['lock'].acquire()
                                self.stage_dataflow_buffer[k]['data'].append(output_dict)
                                self.stage_dataflow_buffer[k]['lock'].release()
                            end_cp = time.time()
                            self.record_time(k, end_cp - start_cp)
                self.stage_thread_dict[stage_num] = threading.Thread(target=thread_func_factory,
                                                             args=(stage_num, ))

    def start(self):
        for k in sorted(list(self.stage_thread_dict.keys())):
            self.stage_thread_dict[k].start()
        try:
            while True:
                time.sleep(0.1)  # sleep forever
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, exiting...")
            print("Stage\tTotal\tCalls\tMin\tMax\tAvg")
            for k in sorted(list(self.tic_tac_dict.keys())):
                print("{}\t{:.3f}\t{}\t{:.3f}\t{:.3f}\t{:.3f}".format(
                    k,
                    self.tic_tac_dict[k]['total_time'],
                    self.tic_tac_dict[k]['total_calls'],
                    self.tic_tac_dict[k]['min_time'],
                    self.tic_tac_dict[k]['max_time'],
                    self.tic_tac_dict[k]['total_time'] / self.tic_tac_dict[k]['total_calls'],
                ))
            print("Exiting...")
            exit(0)
