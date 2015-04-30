#dont merge in first iteration
                if first_iteration:
                    df_merged = df_curr
                    df_curr = pd.DataFrame(data=np.zeros((0, 2), columns=row[6].partition(' ')[0]))
                    first_iteration = False
                #merge and create empty df for following sector
                else:
                    #add only zip and all establishments data
                    if row[1] != curr_zip:
                        df_curr = df_curr = df_curr.append({'ZIP': row[1]}, ignore_index='true')

                         if first_iteration:
                    df_curr = pd.DataFrame(data=np.zeros((0, 2)), columns=['ZIP', row[6].partition(' ')[0]])
                    first_iteration = False
                if row[6] == curr_sector:
                    if row[1] != curr_zip:
                        index += 1
                        #add zip to df_curr
                        df_curr = df_total.append({'ZIP': row[1]}, ignore_index='true')
                        #add all establishments number
                        df_curr.set_value(index, row[6].partition(' ')[0], row[11])
                    else:
                        index += 1
                else:
                    print df_curr
                    df_total = pd.merge(df_total, df_curr, how='left', left_on='ZIP', right_on='ZIP')
                    df_curr = pd.DataFrame(data=np.zeros((0, 2)), columns=['ZIP', row[6].partition(' ')[0]])
                    df_curr = df_total.append({'ZIP': row[1]}, ignore_index='true')
                    df_curr.set_value(index, row[6].partition(' ')[0], row[11])


             X = StandardScaler().fit_transform(data)
    db = DBSCAN(eps=0.8, min_samples=40).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()