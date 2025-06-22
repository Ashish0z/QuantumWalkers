from automan.api import Problem, Simulation, Automator
import os
import yaml

class QMLBenchmark(Problem):
    def get_name(self): return "qml_benchmark"

    def setup(self):
        with open("config.yaml") as f:
            base_config = yaml.safe_load(f)

        self.cases = []
        for train_size in [50, 100, 200]:
                config = base_config.copy()
                config['train_size'] = train_size
                outdir = os.path.join(self.out_dir, f"ts_{train_size}")
                os.makedirs(outdir, exist_ok=True)
                config_path = os.path.join(outdir, "config.yaml")
                with open(config_path, "w") as f:
                    yaml.dump(config, f)

                self.cases.append(
                    Simulation(outdir, f"python experiments/exp1.py --config {config_path}")
                )

    def run(self):
        return super().run()

if __name__ == "__main__":
    Automator(simulation_dir="simulations", output_dir="results", all_problems=[QMLBenchmark]).run()