import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def log_config(config, output_dir):
    current_time = time.strftime("%Y%m%d-%H%M%S")
    config_file = f"{output_dir}/config_{current_time}.log"
    with open(config_file, 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")    
    logging.info(f"Configuration logged to {config_file}")


def log_metrics(metrics, output_dir):
    current_time = time.strftime("%Y%m%d-%H%M%S")
    metrics_file = f"{output_dir}/metrics_{current_time}.log"
    with open(metrics_file, 'w') as f:
        for model, metric in metrics.items():
            f.write(f"Model: {model}\n")
            for key, value in metric.items():
                if isinstance(value, list):
                    value = ', '.join(map(str, value))
                f.write(f"{key}: {value}\n")
            f.write("\n")
    logging.info(f"Metrics logged to {metrics_file}")


def log_time(total_time, output_dir):
    current_time = time.strftime("%Y%m%d-%H%M%S")
    time_file = f"{output_dir}/time_{current_time}.log"
    with open(time_file, 'w') as f:
        f.write(f"Total execution time: {total_time:.2f} seconds\n")
    logging.info(f"Execution time logged to {time_file}")