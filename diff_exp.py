



import pandas as pd
from scipy.stats import f_oneway
from natsort import natsorted
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

# Example: Loading expression data
# Ensure your data is in a CSV file with rows as genes and columns as samples.
# Replace 'expression_data.csv' with your actual data file path.
# data = pd.read_csv('expression_data.csv', index_col=0)


columns = natsorted(sxdf.columns)
data = sxdf[columns]
data=data[(data.max(axis=1)-data.min(axis=1))>10]
data = np.log2(data+1)

# Define groups
group1 = data.iloc[:, 0:3]  # Samples 1-3
group2 = data.iloc[:, 3:6]  # Samples 4-6
group3 = data.iloc[:, 6:9]  # Samples 7-9
group4 = data.iloc[:, 9:12]  # Samples 10-12

# Perform ANOVA test for each gene
anova_results = []
for gene in data.index:
    stat, p_value = f_oneway(group1.loc[gene], group2.loc[gene], group3.loc[gene], group4.loc[gene])
    anova_results.append((gene, stat, p_value))

# Convert results to a DataFrame
anova_df = pd.DataFrame(anova_results, columns=['Gene', 'F-statistic', 'p-value'])

# Adjust for multiple testing using Benjamini-Hochberg (optional)
from statsmodels.stats.multitest import multipletests
anova_df['Adjusted p-value'] = multipletests(anova_df['p-value'], method='fdr_bh')[1]

# Filter significant genes (e.g., adjusted p-value < 0.05)
significant_genes = anova_df[anova_df['Adjusted p-value'] < 0.05]
finite_genes = significant_genes[~significant_genes['F-statistic'].isin([float('inf')])]

sorted_genes = finite_genes.sort_values(by='Adjusted p-value', ascending=False)

# Save the results
sorted_genes.to_csv('kegg_differentially_expressed_genes.csv', index=False)

print(f"Found {len(sorted_genes)} differentially expressed genes.")

data = data.loc[sorted_genes['Gene']]
data['sum'] = data.sum(axis=1)
data = data.sort_values('sum', ascending=False)
data.drop(columns=['sum'], inplace=True)
# data = data[['sample1','sample2','sample3','sample7','sample8','sample9','sample4','sample5','sample6','sample10','sample11','sample12']]

step = 32
for start in range(0, len(sorted_genes), step):

    # List of genes in the specific KEGG pathway
    # kegg_genes = sorted_genes['Gene'].values[start:min(len(sorted_genes), start+step)]  # Replace with your KEGG genes

    # Step 2: Subset the data for KEGG pathway genes
    # kegg_expression = data.loc[data.index.isin(kegg_genes)]
    kegg_expression = data.iloc[start:min(len(sorted_genes), start+32)]

    # Step 3: Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(kegg_expression, cmap='viridis', annot=True, fmt='.2f', cbar_kws={'label': 'Expression Level'})
    plt.title('Heatmap of DEGs in KEGG Pathway')
    plt.xlabel('Samples/Conditions')
    plt.ylabel('Genes')
    plt.tight_layout()

    # Save the heatmap
    plt.savefig(f'kegg_pathway_heatmap_{start}.png')
    # plt.show()
    plt.close('all')



    # Step 3: Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(kegg_expression, cmap='viridis', annot=False, fmt='.2f', cbar_kws={'label': 'Expression Level'})
    plt.title('Heatmap of DEGs in KEGG Pathway')
    plt.xlabel('Samples/Conditions')
    plt.ylabel('Genes')
    plt.tight_layout()

    # Save the heatmap
    plt.savefig(f'kegg_pathway_heatmap_{start}_nonumber.png')
    # plt.show()
    plt.close('all')



data1 = data[['sample1','sample2','sample3','sample7','sample8','sample9']]

step = 32
for start in range(0, len(sorted_genes), step):

    # List of genes in the specific KEGG pathway
    # kegg_genes = sorted_genes['Gene'].values[start:min(len(sorted_genes), start+step)]  # Replace with your KEGG genes

    # Step 2: Subset the data for KEGG pathway genes
    # kegg_expression = data.loc[data.index.isin(kegg_genes)]
    kegg_expression = data1.iloc[start:min(len(sorted_genes), start+32)]

    # Step 3: Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(kegg_expression, cmap='viridis', annot=True, fmt='.2f', cbar_kws={'label': 'Expression Level'})
    plt.title('Heatmap of DEGs in KEGG Pathway')
    plt.xlabel('Samples/Conditions')
    plt.ylabel('Genes')
    plt.tight_layout()

    # Save the heatmap
    plt.savefig(f'kegg_pathway_heatmap_group13_{start}.png')
    # plt.show()
    plt.close('all')



    # Step 3: Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(kegg_expression, cmap='viridis', annot=False, fmt='.2f', cbar_kws={'label': 'Expression Level'})
    plt.title('Heatmap of DEGs in KEGG Pathway')
    plt.xlabel('Samples/Conditions')
    plt.ylabel('Genes')
    plt.tight_layout()

    # Save the heatmap
    plt.savefig(f'kegg_pathway_heatmap_group13_{start}_nonumber.png')
    # plt.show()
    plt.close('all')


data1 = data[['sample4','sample5','sample6','sample10','sample11','sample12']]

step = 32
for start in range(0, len(sorted_genes), step):

    # List of genes in the specific KEGG pathway
    # kegg_genes = sorted_genes['Gene'].values[start:min(len(sorted_genes), start+step)]  # Replace with your KEGG genes

    # Step 2: Subset the data for KEGG pathway genes
    # kegg_expression = data.loc[data.index.isin(kegg_genes)]
    kegg_expression = data1.iloc[start:min(len(sorted_genes), start+32)]

    # Step 3: Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(kegg_expression, cmap='viridis', annot=True, fmt='.2f', cbar_kws={'label': 'Expression Level'})
    plt.title('Heatmap of DEGs in KEGG Pathway')
    plt.xlabel('Samples/Conditions')
    plt.ylabel('Genes')
    plt.tight_layout()

    # Save the heatmap
    plt.savefig(f'kegg_pathway_heatmap_group24_{start}.png')
    # plt.show()
    plt.close('all')



    # Step 3: Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(kegg_expression, cmap='viridis', annot=False, fmt='.2f', cbar_kws={'label': 'Expression Level'})
    plt.title('Heatmap of DEGs in KEGG Pathway')
    plt.xlabel('Samples/Conditions')
    plt.ylabel('Genes')
    plt.tight_layout()

    # Save the heatmap
    plt.savefig(f'kegg_pathway_heatmap_group24_{start}_nonumber.png')
    # plt.show()
    plt.close('all')




data1 = data[['sample1','sample2','sample3', 'sample4','sample5','sample6']]

step = 32
for start in range(0, len(sorted_genes), step):

    # List of genes in the specific KEGG pathway
    # kegg_genes = sorted_genes['Gene'].values[start:min(len(sorted_genes), start+step)]  # Replace with your KEGG genes

    # Step 2: Subset the data for KEGG pathway genes
    # kegg_expression = data.loc[data.index.isin(kegg_genes)]
    kegg_expression = data1.iloc[start:min(len(sorted_genes), start+32)]

    # Step 3: Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(kegg_expression, cmap='viridis', annot=True, fmt='.2f', cbar_kws={'label': 'Expression Level'})
    plt.title('Heatmap of DEGs in KEGG Pathway')
    plt.xlabel('Samples/Conditions')
    plt.ylabel('Genes')
    plt.tight_layout()

    # Save the heatmap
    plt.savefig(f'kegg_pathway_heatmap_group12_{start}.png')
    # plt.show()
    plt.close('all')



    # Step 3: Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(kegg_expression, cmap='viridis', annot=False, fmt='.2f', cbar_kws={'label': 'Expression Level'})
    plt.title('Heatmap of DEGs in KEGG Pathway')
    plt.xlabel('Samples/Conditions')
    plt.ylabel('Genes')
    plt.tight_layout()

    # Save the heatmap
    plt.savefig(f'kegg_pathway_heatmap_group12_{start}_nonumber.png')
    # plt.show()
    plt.close('all')






data1 = data[['sample7'  ,  'sample8'   , 'sample9' ,  'sample10'  , 'sample11'   ,'sample12']]

step = 32
for start in range(0, len(sorted_genes), step):

    # List of genes in the specific KEGG pathway
    # kegg_genes = sorted_genes['Gene'].values[start:min(len(sorted_genes), start+step)]  # Replace with your KEGG genes

    # Step 2: Subset the data for KEGG pathway genes
    # kegg_expression = data.loc[data.index.isin(kegg_genes)]
    kegg_expression = data1.iloc[start:min(len(sorted_genes), start+32)]

    # Step 3: Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(kegg_expression, cmap='viridis', annot=True, fmt='.2f', cbar_kws={'label': 'Expression Level'})
    plt.title('Heatmap of DEGs in KEGG Pathway')
    plt.xlabel('Samples/Conditions')
    plt.ylabel('Genes')
    plt.tight_layout()

    # Save the heatmap
    plt.savefig(f'kegg_pathway_heatmap_group34_{start}.png')
    # plt.show()
    plt.close('all')



    # Step 3: Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(kegg_expression, cmap='viridis', annot=False, fmt='.2f', cbar_kws={'label': 'Expression Level'})
    plt.title('Heatmap of DEGs in KEGG Pathway')
    plt.xlabel('Samples/Conditions')
    plt.ylabel('Genes')
    plt.tight_layout()

    # Save the heatmap
    plt.savefig(f'kegg_pathway_heatmap_group34_{start}_nonumber.png')
    # plt.show()
    plt.close('all')
