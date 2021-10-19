from segmind.sys_data import gpu_metrics, system_metrics
from segmind.tracking.fluent import log_metrics
from segmind.utils.logging_utils import try_mlflow_log


def XGBoost_callback(period=1):
    """Create a callback that print evaluation result.
    We print the evaluation results every **period** iterations
    and on the first and the last iterations.
    Parameters
    ----------
    period : int
        The period to log the evaluation results
    show_stdv : bool, optional
         Whether show stdv if provided
    Returns
    -------
    callback : function
        A callback that print evaluation every period iterations.
    """
    def callback(env):
        """internal function."""
        if env.rank != 0 or (not env.evaluation_result_list) or period is False or period == 0:  # noqa: E501
            return
        step = env.iteration

        results = {}
        # gpu_data = gpu_metrics()
        # results.update(gpu_data)

        # Removing system_metrics for now, as these are not frequently used
        # cpu_data = system_metrics()
        # results.update(cpu_data)
        if step % period == 0 or step + 1 == env.begin_iteration or step + 1 == env.end_iteration:  # noqa: E501
            for x in env.evaluation_result_list:
                results[x[0]] = x[1]
            try_mlflow_log(log_metrics, results, step=step)

    return callback
