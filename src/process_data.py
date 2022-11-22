from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd


DISTRICTS = pd.read_csv('data/district.csv', sep=';')
ACCOUNTS = pd.read_csv('data/account.csv', sep=';')
DISP = pd.read_csv('data/disp.csv', sep=';')
CLIENTS = pd.read_csv('data/client.csv', sep=';')


def filter_relevants(trans_dev, account_id, date):
    return trans_dev.loc[trans_dev['account_id'] == account_id].loc[trans_dev['date'] < date]


def count_condition(df, condition):
    return df.loc[condition].shape[0]


def find_credit(subtrans):
    return subtrans.loc[(subtrans['type'] == 'credit') | (subtrans['operation'] == 'credit in cash')]


def find_withdrawal(subtrans):
    return subtrans.loc[(subtrans['type'] == 'withdrawal') | (subtrans['type'] == 'withdrawal in cash') | (subtrans['operation'] == 'withdrawal in cash') | (subtrans['operation'] == 'credit card withdrawal')]


def signed_mean(subtrans):
    return (
        pd.concat([
            find_credit(subtrans)['amount'],
            -find_withdrawal(subtrans)['amount']
        ])
    ).mean()


class ProcessData:

    def __init__(self, cards, loans, trans):
        self.cards = cards
        self.loans = loans
        self.trans = trans

    def get_merged(self):
        df = pd.merge(self.loans, ACCOUNTS, on='account_id', how='left', suffixes=(
            '_loan', '_account'), validate='one_to_one')

        reduced_disp = pd.merge(DISP, CLIENTS, on='client_id', how='left', suffixes=(
            '_disp', '_client'), validate='one_to_one')
        reduced_disp = pd.merge(reduced_disp, self.cards, on='disp_id', how='left', suffixes=(
            '_disp', '_card'), validate='one_to_one')

        reduced_disp['type_card'] = reduced_disp['type_card'].fillna('no card')

        owners = reduced_disp[reduced_disp['type_disp'] == 'OWNER']
        owners.columns = owners.columns.map(
            lambda x: str(x) + '_owner' if x != 'account_id' else x)

        disponents = reduced_disp[reduced_disp['type_disp'] == 'DISPONENT']
        disponents.columns = disponents.columns.map(
            lambda x: str(x) + '_disponent' if x != 'account_id' else x)

        df = pd.merge(df, owners, on='account_id', how='left', suffixes=(
            None, '_something_wrong'), validate='one_to_one')
        df = pd.merge(df, disponents, on='account_id', how='left', suffixes=(
            None, '_something_wrong'), validate='one_to_one')

        return df

    def add_trans_info(self, df):
        rows_trans_dev = [filter_relevants(
            self.trans, row['account_id'], row['date_loan']) for idx, row in df.iterrows()]

        df['count_trans_credits'] = [find_credit(
            subtrans).shape[0] for subtrans in rows_trans_dev]
        df['count_trans_withdrawals'] = [find_withdrawal(
            subtrans).shape[0] for subtrans in rows_trans_dev]
        df['count_trans_credit_cash'] = [count_condition(
            subtrans, (subtrans['operation'] == 'credit in cash')) for subtrans in rows_trans_dev]
        df['count_trans_withdrawal_cash'] = [count_condition(subtrans, (subtrans['operation'] == 'withdrawal in cash') | (
            subtrans['type'] == 'withdrawal in cash')) for subtrans in rows_trans_dev]
        df['count_trans_withdrawal_card'] = [count_condition(
            subtrans, (subtrans['operation'] == 'credit card withdrawal')) for subtrans in rows_trans_dev]
        df['count_trans_collection_other_bank'] = [count_condition(
            subtrans, (subtrans['operation'] == 'collection from another bank')) for subtrans in rows_trans_dev]
        df['count_trans_remittance_other_bank'] = [count_condition(
            subtrans, (subtrans['operation'] == 'remittance to another bank')) for subtrans in rows_trans_dev]
        df['count_trans_ksymbol_interest_credited'] = [count_condition(
            subtrans, (subtrans['k_symbol'] == 'interest credited')) for subtrans in rows_trans_dev]
        df['count_trans_ksymbol_household'] = [count_condition(
            subtrans, (subtrans['k_symbol'] == 'household')) for subtrans in rows_trans_dev]
        df['count_trans_ksymbol_payment_for_statement'] = [count_condition(
            subtrans, (subtrans['k_symbol'] == 'payment for statement')) for subtrans in rows_trans_dev]
        df['count_trans_ksymbol_insurance_payment'] = [count_condition(
            subtrans, (subtrans['k_symbol'] == 'insurance payment')) for subtrans in rows_trans_dev]
        df['count_trans_ksymbol_sanction_interest_if_negative_balance'] = [count_condition(
            subtrans, (subtrans['k_symbol'] == 'sanction interest if negative balance')) for subtrans in rows_trans_dev]
        df['count_trans_ksymbol_oldage_pension'] = [count_condition(
            subtrans, (subtrans['k_symbol'] == 'old-age pension')) for subtrans in rows_trans_dev]

        # the ones below may be NaN
        df['last_trans_balance'] = [subtrans.loc[subtrans['date'] ==
                                                 subtrans['date'].max()]['balance'].values[0] for subtrans in rows_trans_dev]
        df['mean_trans_balance'] = [subtrans['balance'].mean()
                                    for subtrans in rows_trans_dev]
        df['mean_trans_amount_absolute'] = [subtrans['amount'].mean()
                                            for subtrans in rows_trans_dev]
        df['mean_trans_amount_credit'] = [find_credit(
            subtrans)['amount'].mean() for subtrans in rows_trans_dev]
        df['mean_trans_amount_withdrawal'] = [find_withdrawal(
            subtrans)['amount'].mean() for subtrans in rows_trans_dev]
        df['mean_trans_amount_signed'] = [signed_mean(
            subtrans) for subtrans in rows_trans_dev]

        return df

    def drop_cols(self, df):
        df.drop(columns=['account_id', 'disp_id_owner', 'client_id_owner', 'type_disp_owner', 'card_id_owner',
                         'disp_id_disponent', 'client_id_disponent', 'type_disp_disponent', 'card_id_disponent'], inplace=True)
        return df

    def add_sex_and_birthdate(self, df):
        df['owner_male'] = df['birth_number_owner'].apply(lambda x: 0 if int(
            str(x)[2:4]) > 12 else 1)  # TODO check if this is correct
        df['owner_birthdate'] = df['birth_number_owner'].apply(
            lambda x: x-5000 if int(str(x)[2:4]) > 12 else x)
        df['disponent_male'] = df['birth_number_disponent'].apply(
            lambda x: (0 if int(str(x)[2:4]) > 12 else 1) if not pd.isna(x) else x)
        df['disponent_birthdate'] = df['birth_number_disponent'].apply(
            lambda x: (x-5000 if int(str(x)[2:4]) > 12 else x) if not pd.isna(x) else x)
        df.drop(columns=['amount', 'birth_number_owner',
                'birth_number_disponent'], inplace=True)
        return df

    def rename_cols(self, df):
        df.rename(columns={
            'loan_id': 'loan_id',
            'date_loan': 'date_loan',
            'status': 'status',
            'duration': 'duration_loan',
            'payments': 'payments_loan',
            'district_id': 'account_district',
            'frequency': 'account_frequency',
            'date_account': 'account_date',
            'owner_male': 'owner_male',
            'owner_birthdate': 'owner_birthdate',
            'district_id_owner': 'owner_district',
            'type_card_owner': 'owner_card_type',
            'issued_owner': 'owner_card_issued',
            'disponent_male': 'disponent_male',
            'disponent_birthdate': 'disponent_birthdate',
            'district_id_disponent': 'disponent_district',
            'type_card_disponent': 'disponent_card_type',
            'issued_disponent': 'disponent_card_issued',
            'count_trans_credits': 'count_trans_credits',
            'count_trans_withdrawals': 'count_trans_withdrawals',
            'count_trans_credit_cash': 'count_trans_credit_cash',
            'count_trans_withdrawal_cash': 'count_trans_withdrawal_cash',
            'count_trans_withdrawal_card': 'count_trans_withdrawal_card',
            'count_trans_collection_other_bank': 'count_trans_collection_other_bank',
            'count_trans_remittance_other_bank': 'count_trans_remittance_other_bank',
            'count_trans_ksymbol_interest_credited': 'count_trans_ksymbol_interest_credited',
            'count_trans_ksymbol_household': 'count_trans_ksymbol_household',
            'count_trans_ksymbol_payment_for_statement': 'count_trans_ksymbol_payment_for_statement',
            'count_trans_ksymbol_insurance_payment': 'count_trans_ksymbol_insurance_payment',
            'count_trans_ksymbol_sanction_interest_if_negative_balance': 'count_trans_ksymbol_sanction_interest_if_negative_balance',
            'count_trans_ksymbol_oldage_pension': 'count_trans_ksymbol_oldage_pension',
            'last_trans_balance': 'last_trans_balance',
            'mean_trans_balance': 'mean_trans_balance',
            'mean_trans_amount_absolute': 'mean_trans_amount_absolute',
            'mean_trans_amount_credit': 'mean_trans_amount_credit',
            'mean_trans_amount_withdrawal': 'mean_trans_amount_withdrawal',
            'mean_trans_amount_signed': 'mean_trans_amount_signed'
        }, inplace=True)

        return df

    def transform(self):

        df = self.get_merged()
        df = self.add_trans_info(df)
        df = self.drop_cols(df)
        df = self.add_sex_and_birthdate(df)
        df = self.rename_cols(df)

        return df


class DataframeLabelEncoder:

    def __init__(self, df):
        self.df = df

    def fit(self):
        acc_freq_enc = LabelEncoder()
        acc_freq_enc = acc_freq_enc.fit(
            self.df['account_frequency'])

        owner_card_enc = LabelEncoder()
        owner_card_enc = owner_card_enc.fit(
            self.df['owner_card_type'])

        disponent_card_enc = LabelEncoder()
        disponent_card_enc = disponent_card_enc.fit(
            self.df['disponent_card_type'])

        self.encoders = {
            'account_frequency': acc_freq_enc,
            'owner_card_type': owner_card_enc,
            'disponent_card_type': disponent_card_enc
        }

    def transform(self, df):
        df['status'] = df['status'].apply(lambda x: 0 if x == 1 else 1)

        df['account_frequency'] = self.encoders['account_frequency'].transform(
            df['account_frequency'])
        df['owner_card_type'] = self.encoders['owner_card_type'].transform(
            df['owner_card_type'])
        df['disponent_card_type'] = self.encoders['disponent_card_type'].transform(
            df['disponent_card_type'])

        return df
