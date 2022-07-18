def plot_lambda_zeta(richness_path,spt_path):
    f = h5py.File(richness_path, 'r')
    spt_df = pd.read_pickle(spt_path)
    
    halos = f['halos']
    
    column_names = ["gid","R_lambda","lambda"]
    
    richness_df = pd.DataFrame(data=dict(zip(column_names,(halos['gid'],halos['R_lambda'],halos['lambda']))))
    richness_df.set_index("gid",inplace=True)
    
    spt_df.reset_index()
    spt_df.set_index("id",inplace=True)
    
    spt_sel = spt_df[spt_df["SPT_sel"]==1]
    
    df = pd.merge(richness_df,spt_sel,how="inner",right_index=True,left_index=True)
    lambda_binned_df = df.groupby("lambda").mean()
    
    fig=plt.figure()
    
    print("Size of richness catalog:",len(richness_df))
    print("Size of SPT selected catalog:",len(spt_sel))
    print("Size of intersection:",len(df))

    
    plot1 = sns.barplot(lambda_binned_df.index,"zeta",data=lambda_binned_df,hue="SPT_sel")
    plt.show()