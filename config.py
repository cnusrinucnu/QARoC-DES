datasets = {'Adult','Banknote','Blood','Breast','Cardiotocography','Faults','German','Glass','Haberman','Heart','ILPD','Ionosphere','Iris','Laryngeal3','Liver','Magic','Mammographic','Monk2','Pima','Sonar','Statlog','Steel','Thyroid','Transfusion','Vehicle','Vertebral','Voice3','WDVG1','Weaning','Wine'}

datasets = sorted(datasets)

ExperimentPath = "./Experiment/"
NO_classifiers = 100
no_itr = 20
generate_pools = True
do_train = True
do_evaluate = True

methods_names = ['KNORA-U', 'KNORAE', 'DESKNN', 'OLA', 'LCA', 'MLA', 'MCB', 'Rank', 'KNOP', 'META-DES', 'SingleBest', 'Oracle']
NO_techniques = len(methods_names)
OTHERS = True