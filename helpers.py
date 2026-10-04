from imports import *

def save_pool(datasetName,pools):
    path = config.ExperimentPath + "Pools/" + datasetName + '/' + datasetName + "_pools.p"
    poolspec = open(path, mode="wb")
    pickle.dump(pools, poolspec)
    poolspec.close() 

def load_pool(datasetName):
    path = config.ExperimentPath + "Pools/" + datasetName +  '/' + datasetName + "_pools.p"
    poolspec = open(path, mode="rb")
    return pickle.load(poolspec)

def save_model(tec_name,datasetName,ds):
    path = config.ExperimentPath + "Models/" + datasetName +  '/' + tec_name +"_"+datasetName + "_model.p"
    poolspec = open(path, mode="wb")
    pickle.dump(ds, poolspec)
    poolspec.flush()
    poolspec.close()

def load_model(tec_name,datasetName):
    path = config.ExperimentPath + "Models/" + datasetName +  '/' + tec_name +"_"+ datasetName + "_model.p"
    poolspec = open(path, mode="rb")
    return pickle.load(poolspec)

def save_results(tec_name,datasetName,accuracy,labels,yhat):
    path = config.ExperimentPath + "Results/" + datasetName + '/' +tec_name +"_"+ datasetName + "_result.p"
    poolspec = open(path, mode="wb")
    pickle.dump(accuracy, poolspec)
    pickle.dump(labels, poolspec)
    pickle.dump(yhat, poolspec)
    poolspec.flush()
    poolspec.close()

def save_results2(tec_name, datasetName, accuracy, labels, yhat):
    # Save .p file (pickle)
    p_path = os.path.join(config.ExperimentPath, "Results", datasetName, f"{tec_name}_{datasetName}_result.p")
    with open(p_path, mode="wb") as poolspec:
        pickle.dump(accuracy, poolspec)
        pickle.dump(labels, poolspec)
        pickle.dump(yhat, poolspec)
        #print(len(accuracy), len(labels), len(yhat))

    # Save .xlsx file
    df = pd.DataFrame({
        'Accuracy': accuracy,
        'Labels': labels,
        'Predictions': yhat
    })
    # Set the directory where the file will be created  
    directory = os.path.join(config.ExperimentPath, "Results_xls", datasetName)  
    os.makedirs(directory, exist_ok=True)  

    # Set the base filename  
    base_filename = f"{tec_name}_{datasetName}_results.xlsx"  
    base_filename_with_path = os.path.join(directory, base_filename)  

    # Check if the file already exists  
    i = 1  
    new_filename = base_filename_with_path  
    while os.path.exists(new_filename):  
        # Generate a unique filename by appending a number  
        new_filename = os.path.join(directory, f"{os.path.splitext(base_filename)[0]}_{i}{os.path.splitext(base_filename)[1]}")  
        i += 1  

    # Save the DataFrame to the new file  
    df.to_excel(new_filename, index=False, engine='openpyxl')  


def pool_generator(datasetName):
    state = 0
    pools = []
    for itr in range(0, config.no_itr):
        rng = np.random.RandomState(state)
        #[X_train, X_test, X_DSEL, y_train, y_test, y_DSEL] =  np.load('/content/drive/MyDrive/Experiment1/Datasets/' + datasetName +str(itr)+'.npz',allow_pickle=True)
        # Load the data
        try:              
            data = np.load(config.ExperimentPath + 'Datasets/' + datasetName + '/' + datasetName + str(itr) + '.npz', allow_pickle=True)
            X_train = data['X_train']
            X_test = data['X_test']
            X_DSEL = data['X_DSEL']
            y_train = data['y_train']
            y_test = data['y_test']
            y_DSEL = data['y_DSEL']

            learner = Perceptron(max_iter=100, tol=10e-3, alpha=0.001, penalty=None, random_state=rng)
            min_class_count = np.min(np.bincount(y_train.astype(int)))
            safe_cv = max(2, min(5, min_class_count))
            model = CalibratedClassifierCV(learner, cv=safe_cv, method='isotonic')
            pool_classifiers = BaggingClassifier(model, n_estimators=config.NO_classifiers, bootstrap=True, max_samples=1.0, random_state=rng, n_jobs=-1)

            pool_classifiers.fit(X_train,y_train)
            pools.append(pool_classifiers)
        except EOFError as e:
            print(f"Error loading file: {datasetName +  '/' + datasetName + str(itr) + '.npz'}. Skipping...")
            continue
    path = config.ExperimentPath + "/Pools/" + datasetName +  '/' + datasetName + "_pools.p"
    poolspec = open(path, mode="wb")
    pickle.dump(pools, poolspec)

def convert_datasets(datasetName):
    redata = sio.loadmat("./datasets/" + datasetName + ".mat")
    data = redata['dataset']
    X = data[:, 0:-1]
    y = data[:, -1]
    print(datasetName, "is readed.")
    state = 0
    print(datasetName, ': ', X.shape)

    # ### ### ### ### ### ### ### ### ###
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    X[np.isnan(X)] = 0
    # #### **** #### **** #### **** #### **** #### **** #### ****
    # scaler = preprocessing.MinMaxScaler()
    # X = scaler.fit_transform(X)
    # #### **** #### **** #### **** #### **** #### **** #### ****

    yhat = np.zeros((config.no_itr, math.ceil(len(y) / 4)))
    for itr in range(0, config.no_itr):
        # rand = np.random.randint(1,10000,1)
        rng = np.random.RandomState(itr)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y,
                                                            random_state=rng)  # stratify=y
        X_DSEL, X_test, y_DSEL, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test,
                                                          random_state=rng)  # stratify=y_test
        yhat[itr, :] = y_test

        scaler = preprocessing.MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        X_DSEL = scaler.transform(X_DSEL)

        # Save the data
        np.savez( config.ExperimentPath + 'Datasets/' + datasetName + '/' + datasetName + str(itr) + '.npz',
                X_train=X_train, X_test=X_test, X_DSEL=X_DSEL,
                y_train=y_train, y_test=y_test, y_DSEL=y_DSEL)


def collecting_results():
    methods_names = ['DES_MHA', 'ASyDES', 'KNORA-U', 'KNORAE', 'DESKNN', 'OLA', 'LCA', 'MLA', 'MCB', 'Rank', 'KNOP', 'META-DES', 'SingleBest', 'Oracle']

    for d in config.datasets:
        accuracy_dict = {}
        for m in methods_names:
            accuracy_list = []
            path = config.ExperimentPath + "/Results/" + d + "/" + m + "_" + d + "_result.p"
            with open(path, mode="rb") as f:
                accuracy = pickle.load(f)
            accuracy_list.append(accuracy)
            accuracy_dict[m] = np.array(accuracy_list).reshape(-1,)
        df = pd.DataFrame(accuracy_dict)
        # Calculate mean, std, and rank of mean for each column
        mean = df.mean()
        std = df.std()
        rank = mean.rank(ascending=False)
        df.loc['mean'] = mean
        df.loc['std'] = std
        df.loc['rank'] = rank
        # Set the directory where the file will be created  
        directory = os.path.join(config.ExperimentPath, 'Results_xls', d)  
        os.makedirs(directory, exist_ok=True)  

        # Set the base filename  
        base_filename = f"{d}_accuracy_results.xlsx"  
        base_filename_with_path = os.path.join(directory, base_filename)  

        # Check if the file already exists  
        i = 1  
        new_filename = base_filename_with_path  
        while os.path.exists(new_filename):  
            # Generate a unique filename by appending a number  
            new_filename = os.path.join(directory, f"{os.path.splitext(base_filename)[0]}_{i}{os.path.splitext(base_filename)[1]}")  
            i += 1  

        # Save the DataFrame to the new file  
        with pd.ExcelWriter(new_filename) as writer:  
            df.to_excel(writer, index=True)  
        
# ================================= Other DES Methods Helpers ==========================================

# Prepare the DS techniques. Changing k value to 7.

def others_initialize_ds(pool_classifiers, X_DSEL, y_DSEL, k=7):
    knorau = KNORAU(pool_classifiers, k=k)
    kne = KNORAE(pool_classifiers, k=k)
    desknn = DESKNN(pool_classifiers, k=k)
    ola = OLA(pool_classifiers, k=k)
    lca = LCA(pool_classifiers, k=k)
    mla = MLA(pool_classifiers, k=k)
    mcb = MCB(pool_classifiers, k=k)
    rank = Rank(pool_classifiers, k=k)
    knop = KNOP(pool_classifiers, k=k)
    meta = METADES(pool_classifiers, k=k)

    single_best = SingleBest(pool_classifiers, n_jobs=-1)
    oracle = Oracle(pool_classifiers)
    majority_voting = pool_classifiers

    list_ds = [knorau, kne, desknn, ola, lca, mla, mcb, rank, knop, meta, single_best, oracle]
    methods_names = ['KNORA-U', 'KNORAE', 'DESKNN', 'OLA', 'LCA', 'MLA', 'MCB', 'Rank', 'KNOP', 'META-DES', 'SingleBest', 'Oracle']

    # fit the ds techniques
    for ds in list_ds:
        if ds != majority_voting:
            ds.fit(X_DSEL, y_DSEL)

    return list_ds, methods_names

def others_model_setup(datasetName):
    global methods_names
    pools = load_pool(datasetName)
    ds_matrix = []
    for itr in range(config.no_itr):
        pool_classifiers = pools[itr]
        try:
            #[X_train, X_test, X_DSEL, y_train, y_test, y_DSEL] = np.load('/content/drive/MyDrive/Experiment1/Datasets/' + datasetName + str(itr) + '.npy', allow_pickle=True)
            data = np.load('./Experiment/Datasets/' + datasetName + '/' + datasetName + str(itr) + '.npz', allow_pickle=True)
            X_train = data['X_train']
            X_test = data['X_test']
            X_DSEL = data['X_DSEL']
            y_train = data['y_train']
            y_test = data['y_test']
            y_DSEL = data['y_DSEL']
            list_ds, methods_names = others_initialize_ds(pool_classifiers,X_DSEL,y_DSEL)
            ds_matrix.append(list_ds)
        except EOFError as e:
            print(f"Error loading file: {datasetName +  '/' + datasetName + str(itr) + '.npz'}. Skipping...")
            continue
    for tec in range(config.NO_techniques):
        ds_tec = []
        for itr in range(config.no_itr):
            ds_tec.append(ds_matrix[itr][tec])
        save_model(methods_names[tec], datasetName,ds_tec)

def others_evaluate_model(datasetName):
    for tec in range(config.NO_techniques):
        results = []
        labels = []
        yhat = []
        ds_tec = load_model(methods_names[tec], datasetName)
        for itr in range(config.no_itr):
            try:
                #[X_train, X_test, X_DSEL, y_train, y_test, y_DSEL] = np.load('/content/drive/MyDrive/Experiment1/Datasets/' + datasetName + str(itr) + '.npz',  allow_pickle=True)
                data = np.load('./Experiment/Datasets/' + datasetName + '/' + datasetName + str(itr) + '.npz', allow_pickle=True)
                X_train = data['X_train']
                X_test = data['X_test']
                X_DSEL = data['X_DSEL']
                y_train = data['y_train']
                y_test = data['y_test']
                y_DSEL = data['y_DSEL']
                labels.append(y_test)
                results.append(ds_tec[itr].score(X_test, y_test) * 100)
                if methods_names[tec] == 'Oracle':
                    yhat.append(ds_tec[itr].predict(X_test,y_test))
                else:
                    yhat.append(ds_tec[itr].predict(X_test))
            except EOFError as e:
                print(f"Error loading file: {datasetName + str(itr) + '.npz'}. Skipping...")
                continue
        save_results2(methods_names[tec],datasetName,results,labels,yhat)

# ================================= DES-MHA Helpers ==========================================

def des_mha_initialize_ds(pool_classifiers, X_DSEL, y_DSEL, k=20):
    des_mha = DES_MHA(pool_classifiers, k=k)
    # fit the ds techniques
    des_mha.fit(X_DSEL, y_DSEL)
    return des_mha

def des_mha_model_setup(datasetName, k=20):
    pools = load_pool(datasetName)
    des_matrix = []
    for itr in range(config.no_itr):
        t1 = time.time()
        pool_classifiers = pools[itr]
        try:
            print(f"Setup Number {itr} just started")
            data = np.load('./Experiment/Datasets/' + datasetName +  '/' + datasetName + str(itr) + '.npz', allow_pickle=True)
            X_train = data['X_train']
            X_test = data['X_test']
            X_DSEL = data['X_DSEL']
            y_train = data['y_train']
            y_test = data['y_test']
            y_DSEL = data['y_DSEL']
            ds=des_mha_initialize_ds(pool_classifiers,X_DSEL,y_DSEL, k=k)
            des_matrix.append(ds)
            print(f"Setup Number {itr} - Duration = {time.time()-t1}")
        except EOFError as e:
            print(f"Error loading file: {datasetName +  '/' + datasetName + str(itr) + '.npz'}. Skipping...")
            continue
    ds_mha = []
    for itr in range(config.no_itr):
        ds_mha.append(des_matrix[itr])
    save_model("DES_MHA", datasetName,ds_mha)

def des_mha_evaluate_model(datasetName):
    results = []
    labels = []
    yhat = []
    ds_mha = load_model("DES_MHA", datasetName)
    for itr in range(config.no_itr):
        t1 = time.time()
        try:
            print(f"Evaluate Number {itr} just started")
            data = np.load('./Experiment/Datasets/' + datasetName +  '/' + datasetName + str(itr) + '.npz', allow_pickle=True)
            X_train = data['X_train']
            X_test = data['X_test']
            X_DSEL = data['X_DSEL']
            y_train = data['y_train']
            y_test = data['y_test']
            y_DSEL = data['y_DSEL']
            labels.append(y_test)
            score, y_hat = ds_mha[itr].score(X_test, y_test)
            results.append(score * 100)
            yhat.append(y_hat)
            print(f"Evaluation Number {itr} - Duration = {time.time()-t1}")
        except EOFError as e:
            print(f"Error loading file: {datasetName +  '/' + datasetName + str(itr) + '.npz'}. Skipping...")
            continue
    save_results2("DES_MHA", datasetName,results,labels,yhat)
# ================================== ASyDES ==================================

def asydes_initialize_ds(pool_classifiers, X_DSEL, y_DSEL, k_neighbors=20, ensemble_size=10):
    asydes = ASyDES(
        pool_classifiers,
        k_neighbors=k_neighbors,
        ensemble_size=ensemble_size
    )
    asydes.fit(X_DSEL, y_DSEL)
    return asydes


def asydes_model_setup(datasetName, k_neighbors=20, ensemble_size=10):
    pools = load_pool(datasetName)
    asydes_matrix = []

    for itr in range(config.no_itr):
        t1 = time.time()

        pool_classifiers = pools[itr]

        try:
            print(f"ASyDES Setup Number {itr} just started")

            data = np.load(
                './Experiment/Datasets/' + datasetName + '/' +
                datasetName + str(itr) + '.npz',
                allow_pickle=True
            )

            X_DSEL = data['X_DSEL']
            y_DSEL = data['y_DSEL']

            ds = asydes_initialize_ds(
                pool_classifiers,
                X_DSEL,
                y_DSEL,
                k_neighbors=k_neighbors,
                ensemble_size=ensemble_size
            )

            asydes_matrix.append(ds)

            print(
                f"ASyDES Setup Number {itr} - "
                f"Duration = {time.time()-t1}"
            )

        except EOFError:
            print(
                f"Error loading file: "
                f"{datasetName}/{datasetName}{itr}.npz. Skipping..."
            )
            continue

    save_model("ASyDES", datasetName, asydes_matrix)


def asydes_evaluate_model(datasetName):
    results = []
    labels = []
    yhat = []

    asydes_models = load_model("ASyDES", datasetName)

    for itr in range(config.no_itr):
        t1 = time.time()

        try:
            print(f"ASyDES Evaluation Number {itr} just started")

            data = np.load(
                './Experiment/Datasets/' + datasetName + '/' +
                datasetName + str(itr) + '.npz',
                allow_pickle=True
            )

            X_test = data['X_test']
            y_test = data['y_test']

            labels.append(y_test)

            y_hat = asydes_models[itr].predict(X_test)
            score = accuracy_score(y_test, y_hat)

            results.append(score * 100)
            yhat.append(y_hat)

            print(
                f"ASyDES Evaluation Number {itr} - "
                f"Duration = {time.time()-t1}"
            )

        except EOFError:
            print(
                f"Error loading file: "
                f"{datasetName}/{datasetName}{itr}.npz. Skipping..."
            )
            continue

    save_results2(
        "ASyDES",
        datasetName,
        results,
        labels,
        yhat
    )

# ASyDES import
from asydes import ASyDES

from sklearn.metrics import accuracy_score
