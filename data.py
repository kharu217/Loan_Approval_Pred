from torch.utils.data import DataLoader, Dataset
from utils import *

p_home_own = {
    'RENT' : 0,
    'OWN' : 1,
    'MORTGAGE' : 2,
    'OTHER' : 3
}

intent = {
    'EDUCATION' : 0,
    'MEDICAL' : 1,
    'PERSONAL' : 2,
    'VENTURE' : 3,
    'DEBTCONSOLIDATION' : 4,
    'HOMEIMPROVEMENT' : 5
}

grade = {
    'A' : 0,
    'B' : 1,
    'C' : 2,
    'D' : 3,
    'E' : 4,
    'F' : 5,
    'G' : 6
}

class loan_data(Dataset) :
    def __init__(self, data_path) :
        super().__init__()
        dataframe = pd.read_csv(data_path)
        self.feature = torch.from_numpy(dataframe.loc[:, ~dataframe.columns.isin(['id',
                                                                 'person_home_ownership', 
                                                                 'loan_grade','loan_intent', 
                                                                 'cb_person_cred_hist_length',
                                                                 'cb_person_default_on_file', 
                                                                 'loan_status'])].values)

        self.person_home_own = dataframe.loc[:, 'person_home_ownership'].values
        self.person_home_own = torch.nn.functional.one_hot(str_key(self.person_home_own, p_home_own))

        self.loan_intent = dataframe.loc[:, 'loan_intent'].values
        self.loan_intent = torch.nn.functional.one_hot(str_key(self.loan_intent, intent))

        self.loan_grade = dataframe.loc[:, 'loan_grade'].values
        self.loan_grade = torch.nn.functional.one_hot(str_key(self.loan_grade, grade))

        self.feature = np.concatenate((self.feature, self.person_home_own, self.loan_intent, self.loan_grade), axis=1)
        self.label = dataframe.loc[:, 'loan_status'].values

    def __len__(self) :
        return len(self.feature)
    
    def __getitem__(self, index):
        return self.feature[index], torch.tensor([self.label[index]]).float()

train_data = loan_data(datapath)
train_dataload = DataLoader(train_data, shuffle=True, batch_size=128)
