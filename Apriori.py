import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="mlxtend")

#Change HighAbsence column to LowAbsence to make the data more relevant
data = pd.read_excel("AprioriAlgorithm.xlsx")
df = pd.DataFrame(data)

df = df.drop(
    [
        "StudentID",
        "Age",
        "Gender",
        "ParentalEducation",
        "StudyTimeWeekly",
        "Absences",
        "ParentalSupport",
        "GPA",
        "GradeClass",
    ],
    axis=1,
)

ethnicity = {"Caucasian": 0, "AfricanAmerican": 1, "Asian": 2, "Others": 3}

def perform_apriori(df, min_support=0.2, min_confidence=0.5):
    df_bool = df.astype(bool)
    frequent_itemsets = apriori(df_bool, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    #calculating expectation and interest
    total_transactions = len(df)
    for idx, rule in rules.iterrows():
        consequent = list(rule['consequents'])[0]
        expectation = df_bool[consequent].sum() / total_transactions
        rules.at[idx, 'expectation'] = expectation
        rules.at[idx, 'interest'] = abs(rule['confidence'] - expectation)
    
    return rules

for ethnicity_name, ethnicity_code in ethnicity.items():
    print(f"\n{'='*50}")
    print(f"Association Rules for {ethnicity_name}:")
    print(f"{'='*50}")
    
    ethnicity_data = df[df["Ethnicity"] == ethnicity_code].drop("Ethnicity", axis=1)
    print(f"Shape of data for {ethnicity_name}: {ethnicity_data.shape}")
    
    rules = perform_apriori(ethnicity_data)
    
    if rules.empty:
        print("No rules found with the current thresholds.")
    else:
        # Sort rules by interest value and get top 10
        rules_sorted = rules.sort_values("interest", ascending=False).head(10)
        print(rules_sorted[["antecedents", "consequents", "support", "confidence", "expectation", "interest"]])
        
        # Calculate and display mean interest
        mean_interest = rules_sorted['interest'].mean()
        print(f"\nMean Interest (Top 10 Rules): {mean_interest:.4f}")
        
        # Visualize top 5 rules by interest
        plt.figure(figsize=(12, 6))
        top_5_rules = rules_sorted.head()
        plt.bar(range(len(top_5_rules)), top_5_rules['interest'], align='center')
        plt.xticks(range(len(top_5_rules)), [f"{', '.join(list(ant))} -> {', '.join(list(con))}" 
                                             for ant, con in zip(top_5_rules['antecedents'], top_5_rules['consequents'])], 
                   rotation=90, ha='center')
        plt.ylabel('Interest')
        plt.title(f'Top 5 Association Rules for {ethnicity_name} by Interest')
        
        # Add mean interest line and annotation
        plt.axhline(mean_interest, color='r', linestyle='--', linewidth=2)
        plt.text(plt.xlim()[1], mean_interest, f'Mean: {mean_interest:.4f}', 
                 horizontalalignment='right', verticalalignment='bottom', color='r', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{ethnicity_name}_top_rules_by_interest.png')
        plt.close()

print("\nAnalysis complete. Check the output and generated images.")