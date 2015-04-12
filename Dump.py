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