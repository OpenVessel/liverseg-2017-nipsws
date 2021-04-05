import time
import tensorflow as tf
from functools import wraps
import math


def with_time(time_list):
    def runStepWithTime(step):
        @wraps(step)
        def wrapper(*args, **kwargs):
            # run step
            print('Running step: ' + step.__name__ + "\n")
            start_time = time.time()

            step_output = step()

            print('\nDone step: '+ step.__name__)

            ## run time
            total_time = int(time.time() - start_time)
            time_list.append({'name': step.__name__, 'time' :total_time})
            
            floor_var = math.floor(total_time/60)
            mod_var = total_time % 60
            print("\nTime taken: {} seconds or {} minutes {}s to run\n".format(total_time, floor_var, mod_var))

            # reset tf graph for memory purposes
            tf.reset_default_graph()
            return step_output
        return wrapper
    return runStepWithTime