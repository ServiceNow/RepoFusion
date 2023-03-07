import yaml
import sys


def options(args=None):
    '''
    A placeholder for options to be replaced with actual arg parsing later
      meant to be run from a pwd of the repo root or a path to the common
      config specified in args['common_config']
    '''
    common_config = 'codet5_finetune/common_config.yaml'
    experiment_config = None
    if args is not None:
        if isinstance(args, dict):
            if 'common_config' in args:
                common_config = args['common_config']
            if 'experiment_config' in args:
                experiment_config = args['experiment_config']
        else:
            experiment_config = args
    elif len(sys.argv) > 1:
        experiment_config = sys.argv[1]

    print(sys.argv)
    
    with open(common_config) as f:
        data = yaml.safe_load(f)
        
    if experiment_config is not None:
        with open(experiment_config) as f:
            data.update(yaml.safe_load(f))
    
    opt = type('Opt', (), data)()
    return opt
