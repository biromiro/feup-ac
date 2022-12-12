# Business Understanding

- [ ] analysis of requirements with the end user
- [ ] definition of business goals
- [ ] translation of business goals into data mining goals
- [ ] Tools and project management

## Business Objectives

**Descriptive problem**: TODO

**Predictive problem**: Predict whether a loan will end successfully.

- Classification problem
- Target: *status* column of *loan_dev.csv* file (-1 = bad, 1 = good)
- Minority (positive) class: Loan not paid (*status* = -1)
- Evaluation metric: AUC
- Reduce defaulting (minimize FPR), for example FPR < 0.25
- Don't reduce credit card approval (maximize FNR), for example FNR > 0.95


## ~~Assess Situation~~

## Data Mining Goals (TODO)

Parte disto deve estar em Business Objectives, mas não sei o que é o quê

## Project Plan (TODO)

- Methodology: CRISP-DM
- plan: TODO ?????????
- Project Management tools: ??????
- collaboration tools: Github? (idk if this is what they mean)
- Analytics Tools: ?????
- Database Tools: ?????
- other tools (e.g. data cleaning, visualization): ????


    ```json
{
    'model__n_estimators': 100,
    'model__min_samples_split': 3,
    'model__min_samples_leaf': 2, 
    'model__max_features': 'sqrt', 
    'model__max_depth': 9, 
    'model__criterion': 'entropy', 
    'model__bootstrap': True
}
```