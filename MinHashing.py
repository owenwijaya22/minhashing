import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt

data = pd.read_excel("CleanedStudent_performance_data _.xlsx")
df = pd.DataFrame(data)

np.random.seed(11)

#dropping irrelevant datas
df = df.drop(
    [
        "StudentID",
        "Gender",
        "Age",
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

basket_Caucasian = df[df["Ethnicity"] == 0].drop(["Ethnicity"], axis=1)
basket_AfricanAmerican = df[df["Ethnicity"] == 1].drop(["Ethnicity"], axis=1)
basket_Asian = df[df["Ethnicity"] == 2].drop(["Ethnicity"], axis=1)
basket_Others = df[df["Ethnicity"] == 3].drop(["Ethnicity"], axis=1)

basket_Caucasian = basket_Caucasian.transpose()
basket_AfricanAmerican = basket_AfricanAmerican.transpose()
basket_Asian = basket_Asian.transpose()
basket_Others = basket_Others.transpose()

col_Caucasian = basket_Caucasian.sample(axis=1, random_state=np.random.randint(0, 10000)).iloc[:, 0]
col_AfricanAmerican = basket_AfricanAmerican.sample(axis=1, random_state=np.random.randint(0, 10000)).iloc[:, 0]
col_Asian = basket_Asian.sample(axis=1, random_state=np.random.randint(0, 10000)).iloc[:, 0]
col_Others = basket_Others.sample(axis=1, random_state=np.random.randint(0, 10000)).iloc[:, 0]

# Function to calculate naive Jaccard similarity
def jaccard_similarity(set1, set2):
    intersection = np.sum(np.logical_and(set1, set2))
    union = np.sum(np.logical_or(set1, set2))
    return intersection / union if union != 0 else 0

# Calculate and print naive Jaccard similarities
ethnicities = ["Caucasian", "African American", "Asian", "Others"]
columns = [col_Caucasian, col_AfricanAmerican, col_Asian, col_Others]

group_pairs = []
naive_similarities = []
minhash_similarities = []

print("Naive Jaccard Similarities:")
for i in range(len(ethnicities)):
    for j in range(i+1, len(ethnicities)):
        similarity = jaccard_similarity(columns[i], columns[j])
        print(f"Jaccard Similarity ({ethnicities[i]}, {ethnicities[j]}): {similarity:.4f}")
        group_pairs.append(f"{ethnicities[i]}\n{ethnicities[j]}")
        naive_similarities.append(similarity)

print("\n" + "="*50 + "\n")

def minhash_signature(matrix: np.ndarray, num_hash_functions: int) -> np.ndarray:
    num_rows, num_cols = matrix.shape
    signature_matrix = np.full((num_hash_functions, num_cols), np.inf)
    
    for i in range(num_hash_functions):
        permutation = np.random.permutation(num_rows)
        
        for col in range(num_cols):
            for row in permutation:
                if matrix[row, col] == 1:
                    signature_matrix[i, col] = row
                    break
    
    return signature_matrix

def estimate_jaccard_similarity(sig1: np.ndarray, sig2: np.ndarray) -> float:
    return np.mean(sig1 == sig2)

# Convert your Series to numpy arrays and stack them horizontally
matrix = np.column_stack([
    col_Caucasian.values,
    col_AfricanAmerican.values,
    col_Asian.values,
    col_Others.values
])

num_hash_functions = 100

# Generate MinHash signatures
signatures = minhash_signature(matrix, num_hash_functions)

print("MinHash Estimated Jaccard Similarities with 100 different permutations:")
num_ethnicities = len(ethnicities)

for i in range(num_ethnicities):
    for j in range(i+1, num_ethnicities):
        similarity = estimate_jaccard_similarity(signatures[:, i], signatures[:, j])
        print(f"Estimated Jaccard Similarity ({ethnicities[i]}, {ethnicities[j]}): {similarity:.4f}")
        minhash_similarities.append(similarity)

# Set the style
plt.style.use('ggplot')

# Create the horizontal bar chart
fig, ax = plt.subplots(figsize=(12, 8))

y = np.arange(len(group_pairs))
height = 0.35

rects1 = ax.barh(y - height/2, naive_similarities, height, label='Naive Jaccard', color='skyblue', alpha=0.8)
rects2 = ax.barh(y + height/2, minhash_similarities, height, label='MinHash Estimated', color='lightgreen', alpha=0.8)

ax.set_xlabel('Jaccard Similarity', fontsize=12)
ax.set_title('Comparison of Naive and MinHash Estimated Jaccard Similarities', fontsize=16, fontweight='bold')
ax.set_yticks(y)
ax.set_yticklabels(group_pairs, fontsize=10)
ax.legend(fontsize=10)

# Add value labels on the bars
def add_value_labels(rects):
    for rect in rects:
        width = rect.get_width()
        ax.text(width, rect.get_y() + rect.get_height()/2., f'{width:.3f}',
                ha='left', va='center', fontsize=9, fontweight='bold', 
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

add_value_labels(rects1)
add_value_labels(rects2)

# Add a light grid
ax.grid(axis='x', linestyle='--', alpha=0.7)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Adjust layout and save
fig.tight_layout()
plt.savefig('jaccard_similarities_chart.png', dpi=300, bbox_inches='tight')
print("\nImproved chart has been saved as 'jaccard_similarities_chart.png'")

plt.show()