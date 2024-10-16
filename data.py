from torch.utils.data import DataLoader, Dataset
from utils import *

p_home_own = {
    'RENT' : 1,
    'OWN' : 2,
    'MORTGAGE' : 3,
    'OTHER' : 4
}

intent = {
    'EDUCATION' : 1,
    'MEDICAL' : 2,
    'PERSONAL' : 3,
    'VENTURE' : 4,
    'DEBTCONSOLIDATION' : 5,
    'HOMEIMPROVEMENT' : 6
}

grade = {
    'A' : 1,
    'B' : 2,
    'C' : 3,
    'D' : 4,
    'E' : 5,
    'F' : 6,
    'G' : 7
}

class loan_data(Dataset) :
    def __init__(self, data_path) :
        super().__init__()
        dataframe = pd.read_csv(data_path)
        self.feature = dataframe.loc[:, ~dataframe.columns.isin(['id',
                                                                 'person_home_ownership', 
                                                                 'loan_grade','loan_intent', 
                                                                 'cb_person_cred_hist_length',
                                                                 'cb_person_default_on_file', 
                                                                 'loan_status'])].values
        
        self.person_home_own = dataframe.loc[:, 'person_home_ownership'].values
        self.person_home_own = np.transpose(str_key(self.person_home_own, p_home_own))

        self.loan_intent = dataframe.loc[:, 'loan_intent'].values
        self.loan_intent = np.transpose(str_key(self.loan_intent, intent))

        self.loan_grade = dataframe.loc[:, 'loan_grade'].values
        self.loan_grade = np.transpose(str_key(self.loan_grade, grade))

        self.feature = np.concatenate((self.feature, self.person_home_own, self.loan_intent, self.loan_grade), axis=1)
        self.label = dataframe.loc[:, 'loan_status'].values

    def __len__(self) :
        return len(self.feature)
    
    def __getitem__(self, index):
        return self.feature[index]., torch.tensor([self.label[index]]).float()

train_data = loan_data(datapath)
train_dataload = DataLoader(train_data, shuffle=False, batch_size=16)
