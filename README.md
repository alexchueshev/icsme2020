## Expanding the Number of Reviewers in Open-Source Projects by Recommending Appropriate Developers
Aleksandr Chueshev, Julia Lawall, Reda Bendraou, and Tewfik Ziadi

The 36th IEEE International Conference on Software Maintenance 
and Evolution (ICSME'2020), Adelaide, Australia

## About
A recommender-based approach for OSS projects to
recommend regular reviewers and to expand their number
from among the appropriate developers. 

This study also provides a rich collection of review and development data, 
including information about reviewers, developers and their
commits within five large ASF projects and four Gerrit communities.
_For download, please visit [10.5281/zenodo.3998437](https://doi.org/10.5281/zenodo.3998437)._

## Usage example
Set up `virtualenv`, Python >= 3.6. In the current implementation, the recommender system supports 6 different commands:
- `gs`, `grid` - tune the ALS model hyperparameters using Grid search and cross-validation
- `bs`, `bayes` - tune the ALS model hyperparameters using Bayesian optimization and cross-validation
- `t`, `train` - train an ALS model over a predefined set of hyperparameters
- `b`, `build` - build a multilayer model suitable for making recommendations (see `src/layers`)
- `r`, `recommend` - recommend regular and new potential reviewers for a set of pull requests
- `m`, `metrics` -  estimate the top@_k_ prediction accuracy and MRR

Please check `example/kafka.yaml` and `run.sh` for the parameters supported by each command.

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Contact
Aleksandr Chueshev - alexchueshev@gmail.com

