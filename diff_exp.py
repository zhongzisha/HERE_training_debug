



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



import gseapy as gp
import pandas as pd
from gseapy import barplot, dotplot

df = pd.read_csv('kegg_differentially_expressed_genes.csv')
# Assuming you have a list of significantly expressed genes
gene_list = df['Gene'].values.tolist()  # Replace with your gene list

# Run pathway enrichment using KEGG database
enrichr_results = gp.enrichr(gene_list=gene_list,
                             gene_sets='KEGG_2021_Human',
                             organism='Human',  # 'Human' or 'Mouse', etc.
                             # description='pathway_analysis',
                             outdir='enrichr_kegg')
# if you are only intrested in dataframe that enrichr returned, please set outdir=None
enr = gp.enrichr(gene_list=gene_list, # or "./tests/data/gene_list.txt",
                 gene_sets='KEGG_2021_Human',
                 organism='Human', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=None, # don't write to disk
                )

# Load the results
results = pd.read_csv('enrichr_kegg/KEGG_2021_Human.enrichr.reports.txt', sep='\t')
top10 = results.head(10)
print(top10)


from gseapy import barplot, dotplot
# categorical scatterplot
ax = dotplot(enr.results,
              column="Adjusted P-value",
              x='Gene_set', # set x axis, so you could do a multi-sample/library comparsion
              size=10,
              top_term=5,
              figsize=(3,5),
              title = "KEGG",
              xticklabels_rot=45, # rotate xtick labels
              show_ring=True, # set to False to revmove outer ring
              marker='o',
             )
ax = barplot(enr.res2d,title='KEGG_2021_Human', figsize=(4, 5), color='darkred')





import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

columns = natsorted(sxdf.columns)
data = sxdf[columns]
data=data[(data.max(axis=1)-data.min(axis=1))>10]
# data = np.log2(data+1)
df = data

groups = {
    'g1': ['sample1', 'sample2', 'sample3'],
    'g2': ['sample4', 'sample5', 'sample6'],
    'g3': ['sample7', 'sample8', 'sample9'],
    'g4': ['sample10', 'sample11', 'sample12'],
}

for i, (k1, v1) in enumerate(groups.items()):
    for j, (k2, v2) in enumerate(groups.items()):
        if i >= j:
            continue
                
        # Sample DataFrame
        # Assuming 'df' is the gene expression DataFrame with rows as genes and columns as samples
        # Replace 'group1_samples' and 'group2_samples' with your actual sample IDs
        group1_samples = v1
        # group2_samples = ['sample4', 'sample5', 'sample6']  # Replace with your group 2 sample IDs
        group2_samples = v2

        # Calculate mean expression and fold change
        df['mean_group1'] = df[group1_samples].mean(axis=1)
        df['mean_group2'] = df[group2_samples].mean(axis=1)
        df['log2FC'] = np.log2(df['mean_group2'] / df['mean_group1'])

        # Perform t-tests
        _, p_values = ttest_ind(df[group1_samples], df[group2_samples], axis=1)
        df['p_value'] = p_values

        # Convert p-values to -log10(p-value)
        df['-log10(p-value)'] = -np.log10(df['p_value'])

        # Volcano plot
        plt.figure(figsize=(10, 8))
        plt.scatter(df['log2FC'], df['-log10(p-value)'], alpha=0.7, edgecolor='k')

        # Add threshold lines (optional)
        plt.axhline(y=-np.log10(0.05), color='r', linestyle='--', label='p-value = 0.05')
        plt.axvline(x=-1, color='b', linestyle='--', label='log2FC = -1')
        plt.axvline(x=1, color='b', linestyle='--', label='log2FC = 1')

        # Labels and title
        plt.title(f'Volcano Plot ({k1} vs {k2})')
        plt.xlabel('Log2 Fold Change')
        plt.ylabel('-Log10(p-value)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'kegg_volcano_{k1}_{k2}.png')
        plt.close('all')


import seaborn as sns


diff_exp_gene = pd.read_csv('kegg_differentially_expressed_genes.csv')

for i, (k1, v1) in enumerate(groups.items()):
    for j, (k2, v2) in enumerate(groups.items()):
        if i >= j:
            continue
                
        # Sample DataFrame
        # Assuming 'df' is the gene expression DataFrame with rows as genes and columns as samples
        # Replace 'group1_samples' and 'group2_samples' with your actual sample IDs
        group1_samples = v1
        # group2_samples = ['sample4', 'sample5', 'sample6']  # Replace with your group 2 sample IDs
        group2_samples = v2

        # Calculate mean expression and fold change
        df['mean_group1'] = df[group1_samples].mean(axis=1)
        df['mean_group2'] = df[group2_samples].mean(axis=1)
        df['log2FC'] = np.log2(df['mean_group2'] / df['mean_group1'])
        df['gene'] = df.index.values

        # Perform t-tests
        _, p_values = ttest_ind(df[group1_samples], df[group2_samples], axis=1)
        df['p_value'] = p_values

        # Convert p-values to -log10(p-value)
        df['neg_log10_pval'] = -np.log10(df['p_value'])

        # Highlight genes of interest
        df['highlight'] = df.index.isin(diff_exp_gene['Gene'].values[:20])

        # Create the plot
        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            data=df,
            x='log2FC',
            y='neg_log10_pval',
            hue='highlight',  # Highlight specific genes
            palette={True: 'red', False: 'blue'},
            alpha=0.7,
            s=100,
            legend=False
        )

        # Add labels for genes of interest
        for ii, row in df.iterrows():
            if row['highlight']:
                plt.text(row['log2FC'], row['neg_log10_pval'], row['gene'],
                        fontsize=9, ha='right')

        # Add significance thresholds
        plt.axhline(-np.log10(0.05), color='grey', linestyle='--', linewidth=1)  # p-value threshold
        plt.axvline(0, color='grey', linestyle='--', linewidth=1)  # logFC threshold

        # Customize plot
        plt.title("Volcano Plot", fontsize=16)
        plt.xlabel("Log2 Fold Change", fontsize=14)
        plt.ylabel("-Log10(p-value)", fontsize=14)
        # plt.legend(title="Gene Highlight", loc="upper left")
        plt.tight_layout()
        plt.savefig(f'kegg_volcano2_{k1}_{k2}.png')
        plt.close('all')



import pandas as pd
from gseapy import enrichr, barplot

from natsort import natsorted
from matplotlib import pyplot as plt

columns = natsorted(sxdf.columns)
data = sxdf[columns]
data=data[(data.max(axis=1)-data.min(axis=1))>10]
# data = np.log2(data+1)
data['sum'] = data.sum(axis=1)
data = data.sort_values('sum', ascending=False)
data.drop(columns=['sum'], inplace=True)

diff_exp_gene = pd.read_csv('kegg_differentially_expressed_genes.csv')


significant_genes = [v.upper() for v in diff_exp_gene['Gene'].values.tolist()]
library = 'KEGG_2021_Human'

# Perform Enrichr analysis
results = enrichr(gene_list=significant_genes,
                  gene_sets=library,
                  organism="Human",  # Change to "Mouse" or others if needed
                  # outdir="enrichr_results",  # Output directory
                  cutoff=0.05  # Adjust the p-value cutoff
                 )

barplot(results.res2d, title=f"Top 15 enriched pathways", top_term=15, cutoff=0.65, figsize=(12, 7))
plt.tight_layout()
plt.savefig('kegg_enriched.png')
plt.close('all')


