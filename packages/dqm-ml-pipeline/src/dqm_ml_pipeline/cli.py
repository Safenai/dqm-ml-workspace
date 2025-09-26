import argparse
import logging

import yaml

from dqm_ml_pipeline.pipeline import DatasetPipeline

logger = logging.getLogger(__name__)


# TODO get parameters, logs, ... 
def execute():
    parser = argparse.ArgumentParser(prog='dqm-ml',
                                 description='DQM-ML Pipeline client',
                                 epilog='for more informations see README')

    parser.add_argument('command',
                        choices=['process','check'],
                        help='interact with ks')

    parser.add_argument("-p", "--pipeline", type=str, default="pipeline.yaml", help='pipeline file to execute')  # noqa: E501

    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-q', '--quiet', action='store_true')

    args = parser.parse_args()

    if args.verbose:
        pass  # CustomFormatter.init_log(format="%(name)s - %(message)s (%(filename)s:%(lineno)d)", level=logging.DEBUG)  # noqa: E501
    elif args.quiet:
        pass  # CustomFormatter.init_log(format="%(message)s", level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.DEBUG)
        # CustomFormatter.init_log(format="%(message)s", level=logging.INFO)


    with open(args.pipeline) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        
    # TODO : get parameters from config
    pipeline = DatasetPipeline(config=config['pipeline_config'])
    pipeline.run()  


if __name__ == "__main__":
    execute()
