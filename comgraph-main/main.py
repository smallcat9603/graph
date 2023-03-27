from exp import *


def exp_rwer_among_strategies():
    PATHS = [
        ("graph/Email-Enron.txt", "graph/CA-GrQc.txt"),
        ("graph/0.1-0.01-3-100-normalorder.gr",
         "graph/0.3-0.01-3-100-mixedorder.gr"),
        ("graph/socfb-Caltech36.mtx", "graph/web-edu.mtx"),
        ("graph/web-edu.mtx", "graph/CA-GrQc.txt"),
        ("graph/socfb-Caltech36.mtx", "graph/CA-GrQc.txt"),
        ("graph/Email-Enron.txt", "graph/web-edu.mtx")
    ]
    STRATEGIES = [
        "NWF",
        "WAPPR",
        "allocation-rw",
        "dynamic-rw",
        "AllocationRW"
    ]
    RATIOS = [1/1024 * 2 ** (i) for i in range(21)]
    # RATIOS = [1/2, 1, 2]
    NUM_SAMPLES = 1000
    COMPUTE = False
    filenames_rw_stat = []
    filenames_rw_monotonic = []
    filenames_rw_monotonic_ab = []

    for ga_path, gb_path in PATHS:
        ganame = get_gname(ga_path)
        gbname = get_gname(gb_path)
        ga = read_graph(ga_path)
        gb = read_graph(gb_path)
        ndsa = set([nd for nd in ga])
        ndsb = set([nd for nd in gb])
        nodes_in_common = list(ndsa.intersection(ndsb))
        if NUM_SAMPLES and len(nodes_in_common) > NUM_SAMPLES:
            nodes_in_common = random.sample(nodes_in_common, NUM_SAMPLES)
        if COMPUTE:
            eta = ETA()
            for st_idx, strategy in enumerate(STRATEGIES):
                if strategy in ["NWF", "allocation-rw", "dynamic-rw"]:
                    gm = merge_two_graphs(ga, gb, data=False)
                    apprm = APPR(gm)
                for ratio_idx, ratio in enumerate(RATIOS):
                    if strategy == "WAPPR":
                        nga, ngb = normalize_edge_weight(ga, gb, ratio)
                        gm = merge_two_graphs(nga, ngb, data=True)
                        apprm = APPR(gm)
                    elif strategy == "AllocationRW":
                        gm = convert_two_graphs_to_digraph(ga, gb, ratio)
                        apprm = DAPPR(gm)
                    rwersa, rwersb = [], []

                    for seed in nodes_in_common:
                        if strategy in ["NWF", "WAPPR"]:
                            cm, rwera, rwerb = apprm.compute_appr_data(
                                seed, ga, gb)
                        elif strategy == "allocation-rw":
                            cm, rwera, rwerb, _, _ = apprm.compute_allocation_appr(
                                seed, ga, gb, ratio, data=True)
                        elif strategy == "dynamic-rw":
                            cm, rwera, rwerb, _, _ = apprm.compute_dynamic_appr(
                                seed, ga, gb, ratio, data=True)
                        elif strategy == "AllocationRW":
                            # cm, rwera, rwerb, _, _ = apprm.compute_dynamic_appr(
                            #     seed, ga, gb, ratio, data=True)
                            pass
                        rwersa.append(rwera)
                        rwersb.append(rwerb)
                    dict_of_lists = {
                        f"seed-{strategy}-{ratio}": nodes_in_common,
                        f"rwera-{strategy}-{ratio}": rwersa,
                        f"rwerb-{strategy}-{ratio}": rwersb
                    }
                    df = pd.DataFrame(dict_of_lists)
                    filename = f'tmp/rwers-{ganame}-{gbname}-{NUM_SAMPLES}-{strategy}-{ratio}.txt'
                    df.to_csv(filename, index=False)
                print("ETA:", eta.eta((st_idx + 1) / len(STRATEGIES)))
        rwers_statistics = []
        list_errorbars_rwers = []
        bottom = INF
        top = 0
        df_monot_a = pd.DataFrame(
            columns=['Strategy', 'A_monotonic', 'A_non-monotonic'])
        df_monot_b = pd.DataFrame(
            columns=['Strategy', 'B_monotonic', 'B_non-monotonic'])
        df_monot_ab = pd.DataFrame(
            columns=['Strategy', 'AB_monotonic', 'AB_non-monotonic'])
        for strategy in STRATEGIES:
            meds = []
            q1_diffs = []
            q3_diffs = []
            concat_df_a, concat_df_b, concat_df_ab = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            for ratio_idx, ratio in enumerate(RATIOS):
                filename = f'tmp/rwers-{ganame}-{gbname}-{NUM_SAMPLES}-{strategy}-{ratio}.txt'
                df = pd.read_csv(filename)
                concat_df_a = pd.concat(
                    [
                        concat_df_a,
                        df[f'rwera-{strategy}-{ratio}']
                    ],
                    axis=1
                )
                concat_df_b = pd.concat(
                    [
                        concat_df_b,
                        df[f'rwerb-{strategy}-{ratio}']
                    ],
                    axis=1
                )
                df[f'rwerb/rwera-{strategy}-{ratio}'] = \
                    df[f'rwerb-{strategy}-{ratio}'] / \
                    df[f'rwera-{strategy}-{ratio}']
                concat_df_ab = pd.concat(
                    [
                        concat_df_ab,
                        df[f'rwerb/rwera-{strategy}-{ratio}']
                    ],
                    axis=1
                )
                q1, med, q3 = df[f'rwerb/rwera-{strategy}-{ratio}']\
                    .quantile([0.25, 0.5, 0.75])
                meds.append(med)
                q1_diffs.append(med - q1)
                q3_diffs.append(q3 - med)
                bottom = min(bottom, q1)
                top = max(top, q3)
            rwers_statistics.append(meds)
            list_errorbars_rwers.append([q1_diffs, q3_diffs])
            monot_a = frac_rows_monotonoic(concat_df_a)
            monot_b = frac_rows_monotonoic(concat_df_b)
            monot_ab = frac_rows_monotonoic(concat_df_ab)
            df_monot_a.loc[len(df_monot_a.index)] = [
                strategy, monot_a, 1 - monot_a]
            df_monot_b.loc[len(df_monot_b.index)] = [
                strategy, monot_b, 1 - monot_b]
            df_monot_ab.loc[len(df_monot_ab.index)] = [
                strategy, monot_ab, 1 - monot_ab]
        filename = f'tmp/rwers-{ganame}-{gbname}-{NUM_SAMPLES}.pdf'
        draw_chart(
            RATIOS,
            rwers_statistics,
            list_errorbars_rwers,
            labels=STRATEGIES,
            title="[RWer in B] / [Rwer in A] over r",
            x_axis_title="r",
            y_axis_title="[RWer in B] / [Rwer in A]",
            left=-0.01,
            right=1.01,
            top=top * 1.1,
            bottom=bottom / 1.1,
            filename=filename,
            xscale="log",
            yscale="log",
            print_filename=True,
        )
        filenames_rw_stat.append(filename)
        filename = f'tmp/monotonic-rw-{ganame}-{gbname}-{NUM_SAMPLES}.pdf'
        draw_posneg_bar_chart(
            df_monot_a,
            df_monot_b,
            label_name="Strategy",
            filename=filename,
            title=f'%monotonic\n(metric: rwer, A: {ganame}, B: {gbname})',
            print_filename=True,
        )
        filenames_rw_monotonic.append(filename)
        filename = f'tmp/monotonic-ab-rw-{ganame}-{gbname}-{NUM_SAMPLES}.pdf'
        draw_band_chart(
            df_monot_ab,
            label_name="Strategy",
            title=f'% monotonic B/A \n(metric: rwer, A: {ganame}, B: {gbname})',
            loc="lower right",
            filename=filename,
            print_filename=True,
        )
        filenames_rw_monotonic_ab.append(filename)
    filename = concatanate_images(
        filenames_rw_stat,
        f"tmp/rw-stat-{NUM_SAMPLES}",
        3,
        2,
        print_filename=True,
    )
    upload_to_imgbb(filename)
    filename = concatanate_images(
        filenames_rw_monotonic,
        f"tmp/monotonic-{NUM_SAMPLES}",
        3,
        2,
        print_filename=True,
    )
    upload_to_imgbb(filename)
    filename = concatanate_images(
        filenames_rw_monotonic_ab,
        f"tmp/monotonic-ab-{NUM_SAMPLES}",
        3,
        2,
        print_filename=True,
    )
    upload_to_imgbb(filename)


def frac_rows_monotonoic(df: pd.DataFrame) -> float:
    t = df.T
    df_monotonic_increasing = t.apply(lambda x: x.is_monotonic_increasing)
    df_monotonic_decreasing = t.apply(lambda x: x.is_monotonic_decreasing)
    df_monotonic = df_monotonic_increasing | df_monotonic_decreasing
    return df_monotonic.sum() / df_monotonic.count()


def fracs_monotonoic_increasing_pairs(df: pd.DataFrame) -> List[float]:
    t = df.T
    monotonoic_increasings = []
    for idx in df.index:
        column = list(t[idx])
        cnt = 0
        for i in range(len(column) - 1):
            if column[i] <= column[i + 1]:
                cnt += 1
        monotonoic_increasings.append(cnt / (len(column) - 1))
    return monotonoic_increasings


def fracs_monotonoic_decreasing_pairs(df: pd.DataFrame) -> List[float]:
    t = df.T
    monotonoic_decreasings = []
    for idx in df.index:
        column = list(t[idx])
        cnt = 0
        for i in range(len(column) - 1):
            if column[i] >= column[i + 1]:
                cnt += 1
        monotonoic_decreasings.append(cnt / (len(column) - 1))
    return monotonoic_decreasings


def convert_to_meds_and_errors_q1q3(df: pd.DataFrame) -> Tuple[List[float], List[List[float]]]:
    df_q = df.quantile([0.25, 0.5, 0.75])
    df_q.loc['med - q1'] = df_q.loc[0.5] - df_q.loc[0.25]
    df_q.loc['q3 - med'] = df_q.loc[0.75] - df_q.loc[0.5]
    return list(df_q.loc[0.5]), [list(df_q.loc['med - q1']), list(df_q.loc['q3 - med'])]


def convert_to_percentile_data(l: List[float]) -> Tuple[List[float], List[float]]:
    l = l[:]
    l.sort()
    x_axis = []
    y_axis = []
    for i, v in enumerate(l):
        x_axis.append((i + 1) / len(l))
        y_axis.append(v)
    return x_axis, y_axis


def sub_max_min(df: pd.DataFrame) -> List[float]:
    df_max = df.max(axis=1)
    df_min = df.min(axis=1)
    return list(df_max - df_min)


def exp_true_among_strategies():
    PATHS = [
        # ("graph/Email-Enron.txt", "graph/CA-GrQc.txt"),
        # ("graph/0.1-0.01-3-100-normalorder.gr",
        #  "graph/0.3-0.01-3-100-mixedorder.gr"),
        # ("graph/socfb-Caltech36.mtx", "graph/web-edu.mtx"),
        # ("graph/web-edu.mtx", "graph/CA-GrQc.txt"),
        # ("graph/socfb-Caltech36.mtx", "graph/CA-GrQc.txt"),
        # ("graph/Email-Enron.txt", "graph/web-edu.mtx"),
        # ("graph/aucs-lunch.gr", "graph/aucs-facebook.gr"),
        ("graph/Airports-Lufthansa.gr", "graph/Airports-Ryanair.gr"),
        # ("graph/dkpol-ff.gr", "graph/dkpol-Re.gr"),
        ("graph/Rattus-DI.gr",
         "graph/Rattus-PA.gr"),
    ]
    STRATEGIES = [
        "NWF",
        "ClassicRW",
        "RelaxedRW",
        "WAPPR",
        "WAPPRS",

        # OPTIONS
        # "WAPPRS-0.125",
        # "WAPPRS-0.25",
        # "WAPPRS-0.5",
        # "WAPPRS-1",
        # "WAPPRS-2",
        # "WAPPRS-4",
        # "WAPPRS-8",

        # OLD
        # "allocation-rw",
        # "dynamic-rw",
        # "OneBased",
        # "AllocationRW",
        # "AllocationRW-ZB",
        # "DynamicRW",
        # "DynamicRW-ZB",
        # "ML-LCD",
    ]
    NUM_PLOTS = 21
    RATIOS = [1 / LARGE] + [(i + 1) / (NUM_PLOTS + 1)
                            for i in range(NUM_PLOTS)] + [1 - 1 / LARGE]
    filenames_tks_monotonic = []
    # filenames_tks_ab_monotonic = []
    # filenames_tks_ab_med = []
    filenames_tks_a_med = []
    filenames_frac_monot_tks_a_chart = []
    # filenames_frac_monot_tks_ab_chart = []
    filenames_maxmin_tks_a_chart = []
    # filenames_maxmin_tks_ab_chart = []

    for ga_path, gb_path in PATHS:
        ganame = get_gname(ga_path)
        gbname = get_gname(gb_path)
        dash_index = ganame.find("-")
        dname = ganame[:dash_index]
        laname = ganame[dash_index + 1:]
        lbname = gbname[dash_index + 1:]
        ga = read_graph(ga_path)
        gb = read_graph(gb_path)
        cas = read_clusters(f"cluster/{ganame}/appr.txt")
        cbs = read_clusters(f"cluster/{gbname}/appr.txt")
        cas_mllcd = read_clusters(f"cluster/{ganame}/mllcd.txt")
        cbs_mllcd = read_clusters(f"cluster/{gbname}/mllcd.txt")
        df_monot_tks_a = pd.DataFrame(
            columns=['Strategy', 'A_monotonic', 'A_non-monotonic'])
        df_monot_tks_b = pd.DataFrame(
            columns=['Strategy', 'B_monotonic', 'B_non-monotonic'])
        df_monot_tks_ab = pd.DataFrame(
            columns=['Strategy', 'AB_monotonic', 'AB_non-monotonic'])
        list_meds_tks_a, list_errs_tks_a = [], []
        list_meds_tks_b, list_errs_tks_b = [], []
        list_meds_tks_ab, list_errs_tks_ab = [], []
        list_x_frac_monot_tks_a, list_y_frac_monot_tks_a = [], []
        list_x_frac_monot_tks_b, list_y_frac_monot_tks_b = [], []
        list_x_frac_monot_tks_ab, list_y_frac_monot_tks_ab = [], []
        list_x_maxmin_tks_a, list_y_maxmin_tks_a = [], []
        list_x_maxmin_tks_b, list_y_maxmin_tks_b = [], []
        list_x_maxmin_tks_ab, list_y_maxmin_tks_ab = [], []
        for strategy in STRATEGIES:
            data_tks_a = {ratio: [] for ratio in RATIOS}
            data_tks_b = {ratio: [] for ratio in RATIOS}
            data_tks_ab = {ratio: [] for ratio in RATIOS}
            min_top_sizes = defaultdict(lambda: max(len(ga), len(gb)))
            for ratio in RATIOS:
                cms = read_clusters(
                    f"cluster/{ganame}-{gbname}/{strategy}-{ratio}.txt")
                topks = read_clusters(
                    f"cluster/{ganame}-{gbname}-top/{strategy}-{ratio}.txt",
                    return_type=list,
                )
                l_cms = [(k, v) for k, v in cms.items()]
                l_cms.sort()
                for seed, cm in l_cms:
                    top_size = len(topks[seed])
                    min_top_sizes[seed] = min(min_top_sizes[seed], top_size)
            for ratio in RATIOS:
                topks = read_clusters(
                    f"cluster/{ganame}-{gbname}-top/{strategy}-{ratio}.txt",
                    return_type=list,
                )
                l_topks = [(k, v) for k, v in topks.items()]
                l_topks.sort()
                for i, (seed, topk) in enumerate(l_topks):
                    if strategy == "ML-LCD":
                        ca = cas_mllcd[seed]
                        cb = cbs_mllcd[seed]
                    else:
                        ca = cas[seed]
                        cb = cbs[seed]
                    ca_size = min(len(ca), min_top_sizes[seed])
                    cb_size = min(len(cb), min_top_sizes[seed])
                    top_a = set(topk[:ca_size])
                    top_b = set(topk[:cb_size])
                    tks_a = len(top_a.intersection(ca)) / len(top_a)
                    tks_b = len(top_b.intersection(cb)) / len(top_b)
                    data_tks_a[ratio].append(tks_a)
                    data_tks_b[ratio].append(tks_b)
                    if tks_a != 0:
                        data_tks_ab[ratio].append(tks_b / tks_a)
                    else:
                        data_tks_ab[ratio].append(max(len(ga), len(gb)))
            df_tks_a = pd.DataFrame(data_tks_a).loc[:, RATIOS]
            df_tks_b = pd.DataFrame(data_tks_b).loc[:, RATIOS]
            df_tks_ab = pd.DataFrame(data_tks_ab).loc[:, RATIOS]
            meds_tks_a, errs_tks_a = convert_to_meds_and_errors_q1q3(df_tks_a)
            meds_tks_b, errs_tks_b = convert_to_meds_and_errors_q1q3(df_tks_b)
            meds_tks_ab, errs_tks_ab = convert_to_meds_and_errors_q1q3(
                df_tks_ab)
            list_meds_tks_a.append(meds_tks_a)
            list_errs_tks_a.append(errs_tks_a)
            list_meds_tks_b.append(meds_tks_b)
            list_errs_tks_b.append(errs_tks_b)
            list_meds_tks_ab.append(meds_tks_ab)
            list_errs_tks_ab.append(errs_tks_ab)
            monot_tks_a = frac_rows_monotonoic(df_tks_a)
            monot_tks_b = frac_rows_monotonoic(df_tks_b)
            monot_tks_ab = frac_rows_monotonoic(df_tks_ab)
            df_monot_tks_a.loc[len(df_monot_tks_a.index)] = [
                strategy, monot_tks_a, 1 - monot_tks_a]
            df_monot_tks_b.loc[len(df_monot_tks_b.index)] = [
                strategy, monot_tks_b, 1 - monot_tks_b]
            df_monot_tks_ab.loc[len(df_monot_tks_ab.index)] = [
                strategy, monot_tks_ab, 1 - monot_tks_ab]
            l_monotonic_a = fracs_monotonoic_increasing_pairs(df_tks_a)
            l_monotonic_b = fracs_monotonoic_decreasing_pairs(df_tks_b)
            l_monotonic_ab = fracs_monotonoic_decreasing_pairs(df_tks_ab)
            x_frac_monot_tks_a, y_frac_monot_tks_a = convert_to_percentile_data(
                l_monotonic_a)
            x_frac_monot_tks_b, y_frac_monot_tks_b = convert_to_percentile_data(
                l_monotonic_b)
            x_frac_monot_tks_ab, y_frac_monot_tks_ab = convert_to_percentile_data(
                l_monotonic_ab)
            list_x_frac_monot_tks_a.append(x_frac_monot_tks_a)
            list_y_frac_monot_tks_a.append(y_frac_monot_tks_a)
            list_x_frac_monot_tks_b.append(x_frac_monot_tks_b)
            list_y_frac_monot_tks_b.append(y_frac_monot_tks_b)
            list_x_frac_monot_tks_ab.append(x_frac_monot_tks_ab)
            list_y_frac_monot_tks_ab.append(y_frac_monot_tks_ab)

            l_maxmin_a = sub_max_min(df_tks_a)
            l_maxmin_b = sub_max_min(df_tks_b)
            l_maxmin_ab = sub_max_min(df_tks_ab)
            x_maxmin_tks_a, y_maxmin_tks_a = convert_to_percentile_data(
                l_maxmin_a)
            x_maxmin_tks_b, y_maxmin_tks_b = convert_to_percentile_data(
                l_maxmin_b)
            x_maxmin_tks_ab, y_maxmin_tks_ab = convert_to_percentile_data(
                l_maxmin_ab)
            list_x_maxmin_tks_a.append(x_maxmin_tks_a)
            list_y_maxmin_tks_a.append(y_maxmin_tks_a)
            list_x_maxmin_tks_b.append(x_maxmin_tks_b)
            list_y_maxmin_tks_b.append(y_maxmin_tks_b)
            list_x_maxmin_tks_ab.append(x_maxmin_tks_ab)
            list_y_maxmin_tks_ab.append(y_maxmin_tks_ab)

        filename = f'tmp/med-tks-a-{ganame}-{gbname}.pdf'
        draw_chart(
            [RATIOS] * len(list_meds_tks_a),
            list_meds_tks_a,
            # list_errorbars=list_errs_tks_a,
            labels=STRATEGIES,
            title=f'{dname} (A: {laname}, B: {lbname})',
            x_axis_title="r",
            y_axis_title="TKS",
            loc="lower right",
            # xscale="log",
            left=-0.01,
            right=1.01,
            top=1.01,
            bottom=-0.01,
            filename=filename,
            print_filename=True,
        )
        filenames_tks_a_med.append(filename)

        filename = f'tmp/med-tks-a-{gbname}-{ganame}.pdf'
        for i, l in enumerate(list_meds_tks_b):
            list_meds_tks_b[i] = l[::-1]
        for i, l in enumerate(list_errs_tks_b):
            list_errs_tks_b[i][0] = l[0][::-1]
            list_errs_tks_b[i][1] = l[1][::-1]
        draw_chart(
            [RATIOS] * len(list_meds_tks_b),
            list_meds_tks_b,
            # list_errorbars=list_errs_tks_b,
            labels=STRATEGIES,
            title=f'{dname} (A: {lbname}, B: {laname})',
            x_axis_title="r",
            y_axis_title="TKS",
            loc="lower right",
            # xscale="log",
            left=-0.01,
            right=1.01,
            top=1.01,
            bottom=-0.01,
            filename=filename,
            print_filename=True,
        )
        filenames_tks_a_med.append(filename)

        filename = f'tmp/frac_monot_tks_a_chart-{ganame}-{gbname}.pdf'
        draw_chart(
            list_x_frac_monot_tks_a,
            list_y_frac_monot_tks_a,
            title=f'{dname} (A: {laname}, B: {lbname})',
            labels=STRATEGIES,
            x_axis_title="percentile",
            y_axis_title="FMI",
            marker=None,
            bottom=-0.01,
            top=1.01,
            left=-0.01,
            right=1.01,
            filename=filename,
            print_filename=True,
        )
        filenames_frac_monot_tks_a_chart.append(filename)

        filename = f'tmp/frac_monot_tks_a_chart-{gbname}-{ganame}.pdf'
        draw_chart(
            list_x_frac_monot_tks_b,
            list_y_frac_monot_tks_b,
            title=f'{dname} (A: {lbname}, B: {laname})',
            labels=STRATEGIES,
            x_axis_title="percentile",
            y_axis_title="FMI",
            marker=None,
            bottom=-0.01,
            top=1.01,
            left=-0.01,
            right=1.01,
            filename=filename,
            print_filename=True,
        )
        filenames_frac_monot_tks_a_chart.append(filename)

        filename = f'tmp/maxmin_tks_a_chart-{ganame}-{gbname}.pdf'
        draw_chart(
            list_x_maxmin_tks_a,
            list_y_maxmin_tks_a,
            title=f'{dname} (A: {laname}, B: {lbname})',
            labels=STRATEGIES,
            x_axis_title="percentile",
            y_axis_title="WR",
            marker=None,
            bottom=-0.01,
            top=1.01,
            left=-0.01,
            right=1.01,
            filename=filename,
            print_filename=True,
        )
        filenames_maxmin_tks_a_chart.append(filename)

        filename = f'tmp/maxmin_tks_a_chart-{gbname}-{ganame}.pdf'
        draw_chart(
            list_x_maxmin_tks_b,
            list_y_maxmin_tks_b,
            title=f'{dname} (A: {lbname}, B: {laname})',
            labels=STRATEGIES,
            x_axis_title="percentile",
            y_axis_title="WR",
            marker=None,
            bottom=-0.01,
            top=1.01,
            left=-0.01,
            right=1.01,
            filename=filename,
            print_filename=True,
        )
        filenames_maxmin_tks_a_chart.append(filename)

    # filename = concatanate_images(
    #     filenames_tks_a_med,
    #     f"tmp/med-tks-a",
    #     2,
    #     2,
    #     print_filename=True,
    # )
    # upload_to_imgbb(filename)

    # filename = concatanate_images(
    #     filenames_frac_monot_tks_a_chart,
    #     f"tmp/frac_monot_tks_a_chart",
    #     2,
    #     2,
    #     print_filename=True,
    # )
    # upload_to_imgbb(filename)

    # filename = concatanate_images(
    #     filenames_maxmin_tks_a_chart,
    #     f"tmp/maxmin_tks_a_chart",
    #     2,
    #     2,
    #     print_filename=True,
    # )
    # upload_to_imgbb(filename)


def exp_nodes_in_out_of_topk():
    PATHS = [
        # ("graph/Email-Enron.txt", "graph/CA-GrQc.txt"),
        # ("graph/0.1-0.01-3-100-normalorder.gr",
        #  "graph/0.3-0.01-3-100-mixedorder.gr"),
        # ("graph/socfb-Caltech36.mtx", "graph/web-edu.mtx"),
        # ("graph/web-edu.mtx", "graph/CA-GrQc.txt"),
        # ("graph/socfb-Caltech36.mtx", "graph/CA-GrQc.txt"),
        # ("graph/Email-Enron.txt", "graph/web-edu.mtx")
        # ("graph/aucs-lunch.gr", "graph/aucs-facebook.gr"),
        # ("graph/Airports-Lufthansa.gr", "graph/Airports-Ryanair.gr"),
        # ("graph/dkpol-ff.gr", "graph/dkpol-Re.gr"),
        ("graph/Rattus-DI.gr",
         "graph/Rattus-PA.gr"),
    ]
    STRATEGIES = [
        "NWF",
        "WAPPR",
        "ClassicRW",
        "RelaxedRW",
        "ML-LCD",
        # "AllocationRW",
        # "AllocationRW-ZB",
        # "DynamicRW",
        # "DynamicRW-ZB",
        # "OneBased",
        "WAPPRS",
    ]
    RATIOS = [1/1024 * 2 ** (i) for i in range(21)]
    flaot2str = {2 ** i: f'1/{2 ** -i}' if i <
                 0 else f'{2 ** i}' for i in range(-10, 11)}

    for ga_path, gb_path in PATHS:
        ganame = get_gname(ga_path)
        gbname = get_gname(gb_path)
        cas = read_clusters(f"cluster/{ganame}/appr.txt")
        cbs = read_clusters(f"cluster/{gbname}/appr.txt")
        filenames_a = []
        filenames_b = []
        for strategy in STRATEGIES:
            # ↓ tks_as[seed][ratio] = set(nodes)
            tks_as: Dict[int, Dict[float, Set[int]]] = defaultdict(dict)
            tks_bs: Dict[int, Dict[float, Set[int]]] = defaultdict(dict)
            min_top_sizes = defaultdict(lambda: float('inf'))
            for ratio in RATIOS:
                cms = read_clusters(
                    f"cluster/{ganame}-{gbname}/{strategy}-{ratio}.txt")
                topks = read_clusters(
                    f"cluster/{ganame}-{gbname}-top/{strategy}-{ratio}.txt",
                    return_type=list,
                )
                l_cms = [(k, v) for k, v in cms.items()]
                l_cms.sort()
                for seed, cm in l_cms:
                    top_size = len(topks[seed])
                    min_top_sizes[seed] = min(min_top_sizes[seed], top_size)
            for ratio in RATIOS:
                topks = read_clusters(
                    f"cluster/{ganame}-{gbname}-top/{strategy}-{ratio}.txt",
                    return_type=list,
                )
                l_topks = [(k, v) for k, v in topks.items()]
                l_topks.sort()
                for i, (seed, topk) in enumerate(l_topks):
                    ca = cas[seed]
                    ca_size = min(len(ca), min_top_sizes[seed])
                    cb = cbs[seed]
                    cb_size = min(len(cb), min_top_sizes[seed])
                    top_a = set(topk[:ca_size])
                    top_b = set(topk[:cb_size])
                    tks_a = top_a.intersection(ca)
                    tks_b = top_b.intersection(cb)
                    tks_as[seed][ratio] = tks_a
                    tks_bs[seed][ratio] = tks_b

            hmap_a = {flaot2str[ratio]: {flaot2str[ratio]: 0 for ratio in RATIOS}
                      for ratio in RATIOS}
            for seed, ratio2cluster in tks_as.items():
                for ratio1, cluster1 in ratio2cluster.items():
                    for ratio2, cluster2 in ratio2cluster.items():
                        if len(cluster2) == 0:
                            continue
                        hmap_a[flaot2str[ratio1]][flaot2str[ratio2]] += len(
                            cluster1.intersection(cluster2)
                        ) / len(cluster2)
            for ratio1, cluster1 in ratio2cluster.items():
                for ratio2, cluster2 in ratio2cluster.items():
                    hmap_a[flaot2str[ratio1]][flaot2str[ratio2]] /= len(tks_as)
            df_a = pd.DataFrame(hmap_a)
            filename = f"tmp/inclusion-tks-{ganame}-{gbname}-{strategy}-a.pdf"
            draw_heatmap(
                df_a,
                title=f"Ave. inclusion ratio in A with strategy: {strategy}\n(A: {ganame}, B: {gbname}, left in bottom)",
                cbar_title="Ave. inclusion ratio",
                x_axis_title=f"user-input ratio",
                y_axis_title=f"user-input ratio",
                annot=False,
                figsize=(math.sqrt(2) * WIDTH * 4 / 5, WIDTH),
                filename=filename,
                print_filename=True,
            )
            filenames_a.append(filename)

            hmap_b = {flaot2str[ratio]: {flaot2str[ratio]: 0 for ratio in RATIOS}
                      for ratio in RATIOS}
            for seed, ratio2cluster in tks_bs.items():
                for ratio1, cluster1 in ratio2cluster.items():
                    for ratio2, cluster2 in ratio2cluster.items():
                        if len(cluster2) == 0:
                            continue
                        hmap_b[flaot2str[ratio1]][flaot2str[ratio2]] += len(
                            cluster1.intersection(cluster2)
                        ) / len(cluster2)
            for ratio1, cluster1 in ratio2cluster.items():
                for ratio2, cluster2 in ratio2cluster.items():
                    hmap_b[flaot2str[ratio1]][flaot2str[ratio2]] /= len(tks_bs)
            df_b = pd.DataFrame(hmap_b)
            filename = f"tmp/inclusion-tks-{ganame}-{gbname}-{strategy}-b.pdf"
            draw_heatmap(
                df_b,
                title=f"Ave. inclusion ratio in B with strategy: {strategy}\n(A: {ganame}, B: {gbname}, left in bottom)",
                cbar_title="Ave. inclusion ratio",
                x_axis_title=f"user-input ratio",
                y_axis_title=f"user-input ratio",
                annot=False,
                figsize=(math.sqrt(2) * WIDTH * 4 / 5, WIDTH),
                filename=filename,
                print_filename=True,
            )
            filenames_b.append(filename)
        filename = concatanate_images(
            filenames_a,
            f"tmp/inclusion-tks-a",
            2,
            3,
            print_filename=True,
        )
        upload_to_imgbb(filename)
        filename = concatanate_images(
            filenames_b,
            f"tmp/inclusion-tks-b",
            2,
            3,
            print_filename=True,
        )
        upload_to_imgbb(filename)


def find_non_motonotonic_increases():
    PATHS = [
        # ("graph/Email-Enron.txt", "graph/CA-GrQc.txt"),
        # ("graph/0.1-0.01-3-100-normalorder.gr",
        #  "graph/0.3-0.01-3-100-mixedorder.gr"),
        ("graph/socfb-Caltech36.mtx", "graph/web-edu.mtx"),
        # ("graph/web-edu.mtx", "graph/CA-GrQc.txt"),
        # ("graph/socfb-Caltech36.mtx", "graph/CA-GrQc.txt"),
        # ("graph/Email-Enron.txt", "graph/web-edu.mtx")
    ]
    STRATEGIES = [
        "NWF",
        "WAPPR",
        "ClassicRW",
        "RelaxedRW",
        "ML-LCD",
        "AllocationRW",
        # "AllocationRW-ZB",
        "DynamicRW",
        # "DynamicRW-ZB",
    ]
    RATIOS = [1/1024 * 2 ** (i) for i in range(21)]

    for ga_path, gb_path in PATHS:
        ganame = get_gname(ga_path)
        gbname = get_gname(gb_path)
        cas = read_clusters(f"cluster/{ganame}/appr.txt")
        cbs = read_clusters(f"cluster/{gbname}/appr.txt")
        for strategy in STRATEGIES:
            min_top_sizes = defaultdict(lambda: float('inf'))
            data_tks_a = {ratio: [] for ratio in RATIOS}
            data_tks_b = {ratio: [] for ratio in RATIOS}
            data_tks_ab = {ratio: [] for ratio in RATIOS}
            min_top_sizes = defaultdict(lambda: float('inf'))
            for ratio in RATIOS:
                topks = read_clusters(
                    f"cluster/{ganame}-{gbname}-top/{strategy}-{ratio}.txt",
                    return_type=list,
                )
                for seed, tops in topks.items():
                    min_top_sizes[seed] = min(min_top_sizes[seed], len(tops))
            records = []
            labels = ["ratio", "tks_a", "tks_b",
                      "tks_b/a", "ca_size", "cb_size", ]
            for ratio in RATIOS:
                topks = read_clusters(
                    f"cluster/{ganame}-{gbname}-top/{strategy}-{ratio}.txt",
                    return_type=list,
                )
                l_topks = [(k, v) for k, v in topks.items()]
                l_topks.sort()
                for seed, topk in l_topks[1:2]:
                    ca = cas[seed]
                    ca_size = min(len(ca), min_top_sizes[seed])
                    cb = cbs[seed]
                    cb_size = min(len(cb), min_top_sizes[seed])
                    top_a = set(topk[:ca_size])
                    top_b = set(topk[:cb_size])
                    tks_a = len(top_a.intersection(ca))
                    tks_b = len(top_b.intersection(cb))
                    data_tks_a[ratio].append(tks_a)
                    data_tks_b[ratio].append(tks_b)
                    if seed == 2:
                        if ratio == 16:
                            print(topk)
                        records.append((
                            ratio,
                            tks_a,
                            tks_b,
                            tks_b/tks_a,
                            ca_size,
                            cb_size
                        ))
                    if tks_a != 0:
                        data_tks_ab[ratio].append(tks_b / tks_a)
                    else:
                        data_tks_ab[ratio].append(INF)
            export_table(records, labels)
            df_ab = pd.DataFrame(data_tks_ab)
            monot_true_ab = frac_rows_monotonoic(df_ab)
            print(monot_true_ab)


def strategy_check():
    ga_path, gb_path = "graph/Email-Enron.txt", "graph/CA-GrQc.txt"
    # strategy = "allocationi-rw"
    strategy = "AllocationRW"
    ganame = get_gname(ga_path)
    gbname = get_gname(gb_path)
    ga = read_graph(ga_path)
    gb = read_graph(gb_path)
    appra = APPR(ga)
    apprb = APPR(gb)
    if strategy == "allocationi-rw":
        gm = merge_two_graphs(ga, gb, False)
        apprm = APPR(gm)
    SEED = 14
    ca = set(appra.compute_appr(SEED))
    cb = set(apprb.compute_appr(SEED))
    RATIOS = [1/1024 * 2 ** (i) for i in range(21)]
    for ratio in RATIOS:
        if strategy == "allocationi-rw":
            cm = apprm.compute_allocation_appr(SEED, ga, gb, ratio)
        if strategy == "AllocationRW":
            gm = convert_two_graphs_to_digraph(ga, gb, ratio)
            apprm = DAPPR(gm)
            cm = apprm.compute_appr(SEED)
        cm = set(cm)
        a_uniques = ca - cb
        b_uniques = cb - ca
        a_trues = a_uniques.intersection(cm)
        b_trues = b_uniques.intersection(cm)
        b_over_a = len(b_trues) / \
            len(a_trues) if len(a_trues) != 0 else INF
        print("ratio: ", ratio, ", b/a: ", b_over_a, sep='')


def compare_two_ratios():
    nd1 = set([2, 11, 9, 7, 10, 4, 3, 8, 6, 305, 2785, 2754, 5, 2778, 128, 457, 213, 585, 76, 254, 283, 468, 79, 596, 302, 253, 366, 359, 512, 2789, 522, 226, 460, 332, 170, 280, 689, 442, 396, 303, 2753, 694, 161, 672, 664, 554, 560, 534, 45, 432, 456, 221, 2783, 30,
              537, 81, 632, 2772, 519, 323, 252, 625, 2892, 339, 92, 538, 864, 865, 866, 867, 868, 869, 870, 871, 872, 932, 933, 1090, 1091, 2817, 157, 551, 406, 2787, 188, 2780, 2784, 125, 151, 2792, 467, 662, 265, 453, 670, 331, 520, 484, 362, 673, 355, 114, 169, 735, 365])
    nd2 = set([2, 11, 9, 7, 10, 8, 305, 4, 6, 3, 5, 2785, 2754, 76, 2778, 128, 213, 457, 254, 302, 283, 468, 79, 253, 2789, 512, 585, 664, 369, 226, 170, 596, 366, 280, 332, 303, 2783, 522, 81, 157, 864, 865, 866, 867, 868, 869, 870, 871, 872, 932, 933, 1090, 1091, 2817,
              460, 689, 161, 2787, 694, 442, 456, 662, 672, 359, 625, 252, 534, 396, 339, 554, 467, 2892, 2753, 102, 713, 92, 188, 13, 519, 362, 151, 2772, 537, 2821, 2823, 2825, 2826, 2827, 2828, 323, 560, 265, 432, 326, 221, 45, 78, 632, 538, 30, 2829, 297, 520, 2905, 225])
    ca = set([2, 359, 585, 99, 438, 457, 596, 522, 213, 460, 394, 406, 432, 456, 396, 366, 442, 560, 689, 332, 45, 283, 79, 253, 30, 632, 694, 566, 512, 672, 170, 226, 554, 537, 161, 40, 507, 519, 280, 323, 743, 712, 534, 383, 538, 518, 114, 94, 540, 252, 670, 523,
             484, 577, 339, 420, 112, 735, 164, 453, 536, 188, 435, 626, 303, 174, 67, 468, 288, 638, 377, 355, 151, 163, 92, 265, 122, 91, 124, 108, 648, 90, 28, 704, 370, 107, 401, 194, 21, 201, 148, 85, 365, 573, 627, 278, 510, 356, 207, 470, 369, 351, 224, 762, 734])
    nd2rank = {
        2: 1,
        359: 2,
        585: 3,
        99: 4,
        438: 5,
        457: 6,
        596: 7,
        522: 8,
        213: 9,
        460: 10,
        394: 11,
        406: 12,
        432: 13,
        456: 14,
        396: 15,
        366: 16,
        442: 17,
        560: 18,
        689: 19,
        332: 20,
        45: 21,
        283: 22,
        79: 23,
        253: 24,
        30: 25,
        632: 26,
        694: 27,
        566: 28,
        512: 29,
        672: 30,
        170: 31,
        226: 32,
        554: 33,
        537: 34,
        161: 35,
        40: 36,
        507: 37,
        519: 38,
        280: 39,
        323: 40,
        743: 41,
        712: 42,
        534: 43,
        383: 44,
        538: 45,
        518: 46,
        114: 47,
        94: 48,
        540: 49,
        252: 50,
        670: 51,
        523: 52,
        484: 53,
        577: 54,
        339: 55,
        420: 56,
        112: 57,
        735: 58,
        164: 59,
        453: 60,
        536: 61,
        188: 62,
        435: 63,
        626: 64,
        303: 65,
        174: 66,
        67: 67,
        468: 68,
        288: 69,
        638: 70,
        377: 71,
        355: 72,
        151: 73,
        163: 74,
        92: 75,
        265: 76,
        122: 77,
        91: 78,
        124: 79,
        108: 80,
        648: 81,
        90: 82,
        28: 83,
        704: 84,
        370: 85,
        107: 86,
        401: 87,
        194: 88,
        21: 89,
        201: 90,
        148: 91,
        85: 92,
        365: 93,
        573: 94,
        627: 95,
        278: 96,
        510: 97,
        356: 98,
        207: 99,
        470: 100,
        369: 101,
        351: 102,
        224: 103,
        762: 104,
        734: 105,
    }
    print("- intersection:", nd1.intersection(nd2))
    print("- only in 1: ", end='')
    for nd in (nd1 - nd2):
        if nd in ca:
            print("<ins>", end="")
        print(nd, end="")
        if nd in ca:
            print("</ins>", end="")
        else:
            if nd in nd2rank:
                print(f"({nd2rank[nd]})", end='')
            else:
                print(" (out)", end='')
        print(" ", end='')
    print()
    print("- only in 2: ", end='')
    for nd in (nd2 - nd1):
        if nd in ca:
            print("<ins>", end="")
        print(nd, end="")
        if nd in ca:
            print("</ins>", end="")
        else:
            if nd in nd2rank:
                print(f"({nd2rank[nd]})", end='')
            else:
                print(" (out)", end='')
        print(", ", end='')
    print()


def compute_apprs():
    PATHS = [
        # "graph/Email-Enron.txt",
        # "graph/0.1-0.01-3-100-normalorder.gr",
        # "graph/0.3-0.01-3-100-mixedorder.gr",
        # "graph/CA-GrQc.txt",
        # "graph/web-edu.mtx",
        # "graph/socfb-Caltech36.mtx",
        # "graph/p2p-Gnutella09.txt",
        # "graph/Wiki-Vote.txt",
        "graph/aucs-lunch.gr",
        "graph/aucs-facebook.gr",
        "graph/Airports-Lufthansa.gr",
        "graph/Airports-Ryanair.gr",
        "graph/dkpol-ff.gr",
        "graph/dkpol-Re.gr",
        "graph/Rattus-DI.gr",
        "graph/Rattus-PA.gr",
    ]
    for path in PATHS:
        compute_appr(path)


def conductance_distribution():
    PATHS = [
        "graph/Email-Enron.txt",
        "graph/0.1-0.01-3-100-normalorder.gr",
        "graph/0.3-0.01-3-100-mixedorder.gr",
        "graph/CA-GrQc.txt",
        "graph/web-edu.mtx",
        "graph/socfb-Caltech36.mtx",
        "graph/p2p-Gnutella09.txt",
        "graph/Wiki-Vote.txt",
    ]
    data_conductances = []
    gnames = []
    fnames = []
    ONLY_BETTER_CLUSTERS = False
    DEGREE_NORMALIZED = False

    for path in PATHS:
        g = read_graph(path)
        gname = get_gname(path)
        gnames.append(gname)
        conductances = []
        if DEGREE_NORMALIZED:
            cs = read_clusters(f"cluster/{gname}/appr.txt")
            tops = read_clusters(f"cluster/{gname}/appr-top.txt")
            list_conductances = read_clusters(
                f"cluster/{gname}/appr-conductance.txt", return_type=list, val_type=float)
        else:
            cs = read_clusters(f"cluster/{gname}/appr-ndn.txt")
            tops = read_clusters(f"cluster/{gname}/appr-top-ndn.txt")
            list_conductances = read_clusters(
                f"cluster/{gname}/appr-conductance-ndn.txt", return_type=list, val_type=float)
        cnt_same = 0
        for seed, conds in list_conductances.items():
            csize = len(cs[seed])
            cond = conds[csize - 1]
            if csize == len(tops[seed]):
                cnt_same += 1
            if ONLY_BETTER_CLUSTERS and csize == len(tops[seed]):
                pass
            else:
                conductances.append(cond)
        data_conductances.append(conductances)
        print("size:", len(conductances))
        fname = f"tmp/{gname}-same.pdf"
        draw_pie_chart(
            [cnt_same / len(list_conductances),
             (len(list_conductances) - cnt_same) / len(list_conductances)],
            labels=["same", "different"],
            title=f"% seeds that have the same cluster-size and top-size\n({gname})",
            filename=fname,
            print_filename=True,
        )
        fnames.append(fname)
    filename = "tmp/conductances.pdf"
    draw_boxplot(
        data_conductances,
        labels=gnames,
        y_axis_title="conductance",
        rotation=45,
        bottom=0,
        top=1,
        title=f"conductance distribution (degree_normalized: {DEGREE_NORMALIZED})\n{'(only better clusters)' if ONLY_BETTER_CLUSTERS else ''}",
        filename=filename,
        show_median=False,
    )
    upload_to_imgbb(filename)
    fname = concatanate_images(
        fnames, filename_prefix="tmp/image", num_x=3, num_y=3)
    upload_to_imgbb(fname)


def conductance_distribution_merged():
    PATHS = [
        ("graph/Email-Enron.txt", "graph/CA-GrQc.txt"),
        ("graph/0.1-0.01-3-100-normalorder.gr",
         "graph/0.3-0.01-3-100-mixedorder.gr"),
        ("graph/socfb-Caltech36.mtx", "graph/web-edu.mtx"),
        ("graph/web-edu.mtx", "graph/CA-GrQc.txt"),
        ("graph/socfb-Caltech36.mtx", "graph/CA-GrQc.txt"),
        ("graph/Email-Enron.txt", "graph/web-edu.mtx")
    ]
    strategy = "AllocationRW"
    NORMALIZATIONS = [
        "in",
        "out",
        "in_out"
    ]

    RATIOS = [1/1024 * 2 ** (i) for i in range(21)]
    RATIOS = [1/1024 * 2 ** (i) for i in range(3)]
    fnames = []

    for ga_path, gb_path in PATHS:
        ganame = get_gname(ga_path)
        gbname = get_gname(gb_path)
        list_meds = []
        list_errorbars = []
        for normalization in NORMALIZATIONS:
            ratio2conds = defaultdict(list)
            for ratio in RATIOS:
                path = f"cluster/{ganame}-{gbname}/{strategy}-{ratio}-{normalization}.txt"
                path_conductance = f"cluster/{ganame}-{gbname}-conductance/{strategy}-{ratio}-{normalization}.txt"
                clusters = read_clusters(path)
                conductances = read_clusters(
                    path_conductance, return_type=list, val_type=float)
                list_seed_csize = [(seed, len(c))
                                   for seed, c in clusters.items()]
                list_seed_csize.sort()
                for seed, csize in list_seed_csize:
                    cond = conductances[seed][csize - 1]
                    ratio2conds[ratio].append(cond)
            df = pd.DataFrame(ratio2conds)
            meds, errs = convert_to_meds_and_errors_q1q3(df)
            list_meds.append(meds)
            list_errorbars.append(errs)
        filename = f"tmp/conductance-{ganame}-{gbname}.pdf"
        draw_chart(
            RATIOS,
            list_meds,
            list_errorbars,
            labels=NORMALIZATIONS,
            title=f"conductance over r with 3 normalization degree\n(A: {ganame}, B: {gbname})",
            x_axis_title="r",
            y_axis_title="conductance",
            left=-0.01,
            right=1.01,
            top=1.01,
            bottom=-0.01,
            loc="upper right",
            filename=filename,
            # xscale="log",
            print_filename=True,
        )
        fnames.append(filename)
    concatanate_images(
        fnames, filename_prefix="tmp-conductance", num_x=3, num_y=2)


def draw_ex_graph():
    ga = read_graph("graph/ex_a.gr")
    gb = read_graph("graph/ex_b.gr")
    # ga = read_graph(
    #     "graph/Email-Enron.txt")
    # gb = read_graph("graph/CA-GrQc.txt")
    # ga = read_graph("graph/0.1-0.01-3-100-normalorder.gr")
    # gb = read_graph("graph/0.1-0.01-3-100-normalorder.gr")

    # nga, ngb = normalize_edge_weight(ga, gb, 1024)
    # gm = merge_two_graphs(nga, ngb, data=True)
    # appr = APPR(gm)

    # gm = merge_two_graphs(ga, gb, False)
    gm = convert_two_graphs_to_digraph(ga, gb, 1024, one_based=True)
    appr = DAPPR(gm)

    print(appr.compute_appr(1))
    appr_vec = appr.get_appr_vec()
    for nd in gm:
        try:
            win_degree = sum([gm[nd][nbr]['weight']
                             for nbr in gm.neighbors(nd)])
            wout_degree = sum([gm[nbr][nd]['weight']
                              for nbr in gm.neighbors(nd)])
            print(
                f'{nd}: {appr_vec[nd] / (win_degree + wout_degree)}')
            # print("sum_weight:", (win_degree + wout_degree))
        except AttributeError:
            print(f'{nd}: {appr_vec[nd] / gm.degree(nd)}')
    axes = {
        0: (3, 2),
        1: (5, 1),
        2: (6, 3),
        3: (4, 5),
        4: (2, 4)
    }
    axes = normalize_axes(axes)
    pos = {}
    for i, axis in axes.items():
        pos[i] = axis
    edge_colors = dict()
    edge_colors["blue"] = [e for e in ga.edges()]
    edge_colors["red"] = [e for e in gb.edges()]
    draw_graph(gm, pos=pos, edge_colors=edge_colors)
    # print(gm.edges().data(True))

    # gm = merge_two_graphs(ga, gb, False)
    # apprm = APPR(gm)
    # seed = 0
    # s = time.time()
    # print(apprm.compute_dynamic_weighting_appr(
    #     seed, ga, gb, c=1, one_based=False))
    # print(time.time() - s)
    # s = time.time()
    # print(apprm.compute_dynamic_appr(seed, ga, gb, c=1))
    # print(time.time() - s)


def copy_graph_with_new_ids(
    g: nx.Graph(),
    nodes_to_be_renumbered: List[int],
    addition: int,
) -> nx.Graph:
    g: nx.Graph = copy.deepcopy(g)
    for nd in nodes_to_be_renumbered:
        new_nd = nd + addition
        for nbr in g.neighbors(nd):
            g.add_edge(new_nd, nbr)
        g.remove_node(nd)
    return g


def create_dataset_removing_overlaps():
    PATHS = [
        # ("graph/Email-Enron.txt", "graph/CA-GrQc.txt"),
        # ("graph/0.1-0.01-3-100-normalorder.gr",
        #  "graph/0.3-0.01-3-100-mixedorder.gr"),
        # ("graph/socfb-Caltech36.mtx", "graph/web-edu.mtx"),
        # ("graph/web-edu.mtx", "graph/CA-GrQc.txt"),
        # ("graph/socfb-Caltech36.mtx", "graph/CA-GrQc.txt"),
        # ("graph/Email-Enron.txt", "graph/web-edu.mtx")
        ("graph/aucs-lunch.gr", "graph/aucs-facebook.gr"),
        ("graph/Airports-Lufthansa.gr", "graph/Airports-Ryanair.gr"),
        ("graph/dkpol-ff.gr", "graph/dkpol-Re.gr"),
        ("graph/Rattus-DI.gr",
         "graph/Rattus-PA.gr"),
    ]
    percents = sorted([0.01, 0.05, 0.1, 0.25, 0.5], reverse=True)
    for ga_path, gb_path in PATHS:
        ga = read_graph(ga_path)
        gb = read_graph(gb_path)
        ganame = get_gname(ga_path)
        gbname = get_gname(gb_path)
        nodes_in_common = set(
            [nd for nd in ga]).intersection([nd for nd in gb])
        num_nodes_in_common = len(nodes_in_common)
        nds = list(nodes_in_common)
        max_id = max(max(ga.nodes()), max(gb.nodes()))
        for percent in percents:
            num_samples = int(percent * num_nodes_in_common)
            remaining_nds = random.sample(nds, num_samples)
            nds_to_be_renumbered = set(nds) - set(remaining_nds)
            nds = remaining_nds
            new_gb = copy_graph_with_new_ids(
                gb, nds_to_be_renumbered, max_id)
            gb = new_gb
            max_id += max(gb)

            try:
                dir = f"graph/{ganame}-{gbname}"
                os.makedirs(dir)
            except FileExistsError:
                pass
            path = f"graph/{ganame}-{gbname}/{percent}.gr"
            nx.write_edgelist(gb, path=path, data=False)


def output_merged_clusters_removing_overlaps():
    PATHS = [
        # ("graph/Email-Enron.txt", "graph/CA-GrQc.txt"),
        # ("graph/0.1-0.01-3-100-normalorder.gr",
        #  "graph/0.3-0.01-3-100-mixedorder.gr"),
        # ("graph/web-edu.mtx", "graph/CA-GrQc.txt"),
        # ("graph/socfb-Caltech36.mtx", "graph/CA-GrQc.txt"),
        # ("graph/socfb-Caltech36.mtx", "graph/web-edu.mtx"),
        # ("graph/Email-Enron.txt", "graph/web-edu.mtx"),
        ("graph/aucs-lunch.gr", "graph/aucs-facebook.gr"),
        ("graph/Airports-Lufthansa.gr", "graph/Airports-Ryanair.gr"),
        ("graph/dkpol-ff.gr", "graph/dkpol-Re.gr"),
        ("graph/Rattus-DI.gr",
         "graph/Rattus-PA.gr"),
    ]
    STRATEGIES = [
        # "NWF",
        # "WAPPR",
        # "ClassicRW",
        # "RelaxedRW",
        # "ML-LCD",
        # "WAPPRS",

        # # OLD
        # "AllocationRW",
        # "AllocationRW-ZB",
        # "DynamicRW",
        # "DynamicRW-ZB",
        # "OneBased",
    ]
    percents = [0.01, 0.05, 0.1, 0.25, 0.5]
    SKIP_IF_EXISTS = False
    NUM_PLOTS = 21
    RATIOS = [1 / LARGE] + [(i + 1) / (NUM_PLOTS + 1)
                            for i in range(NUM_PLOTS)] + [1 - 1 / LARGE]
    RATIOS_FOR_ML_LCD = [-1 + 1 / LARGE] + \
        [i / 10 for i in range(-10, 11)] + [1 - 1 / LARGE]

    for ga_path, gb_path in PATHS:
        ganame = get_gname(ga_path)
        gbname = get_gname(gb_path)
        for percent in percents:
            gb_percent_path = f"graph/{ganame}-{gbname}/{percent}.gr"
            ga = read_graph(ga_path)
            gb = read_graph(gb_percent_path)

            nodes_in_common = list(
                set([nd for nd in ga]).intersection([nd for nd in gb]))
            NUM_SAMPLES = None
            if NUM_SAMPLES:
                nodes_in_common = random.sample(nodes_in_common, NUM_SAMPLES)

            try:
                dir = f"cluster/{ganame}-{gbname}"
                os.makedirs(dir)
            except FileExistsError:
                pass
            try:
                dir_top = f"cluster/{ganame}-{gbname}-top"
                os.makedirs(dir_top)
            except FileExistsError:
                pass
            try:
                dir_conductance = f"cluster/{ganame}-{gbname}-conductance"
                os.makedirs(dir_conductance)
            except FileExistsError:
                pass
            eta = ETA()
            for i, strategy in enumerate(STRATEGIES):
                for j, ratio in enumerate(RATIOS):
                    path = f"cluster/{ganame}-{gbname}/{strategy}-{ratio}-removing-{percent}.txt"
                    if SKIP_IF_EXISTS and os.path.isfile(path):
                        continue
                    f = open(path, "w")

                    path_top = f"cluster/{ganame}-{gbname}-top/{strategy}-{ratio}-removing-{percent}.txt"
                    f_top = open(path_top, "w")

                    if strategy != "ML-LCD":
                        path_conductance = f"cluster/{ganame}-{gbname}-conductance/{strategy}-{ratio}.txt"
                        f_conductance = open(path_conductance, "w")

                    if strategy == "WAPPR":
                        nga, ngb = normalize_edge_weight(ga, gb, ratio)
                        gm = merge_two_graphs(nga, ngb, data=True)
                        apprm = APPR(gm)
                    # elif strategy == "AllocationRW":
                    #     gm = convert_two_graphs_to_digraph(ga, gb, ratio)
                    #     apprm = DAPPR(gm)
                    # elif strategy == "AllocationRW-ZB":
                    #     gm = convert_two_graphs_to_digraph(
                    #         ga, gb, ratio, one_based=False)
                    #     apprm = DAPPR(gm)
                    # elif strategy == "OneBased":
                    #     gm = convert_two_graphs_to_digraph_one_based(
                    #         ga, gb, ratio)
                    #     apprm = DAPPR(gm)
                    elif strategy in ["ClassicRW", "RelaxedRW"]:
                        gm = merge_two_graphs(ga, gb, data=False)
                        apprm = APPR(gm, ga, gb)
                    elif strategy == "ML-LCD":
                        mllcd = MLLCD([ga, gb], RATIOS_FOR_ML_LCD[j])
                    elif strategy == "WAPPRS":
                        gm = merge_two_graphs_with_supernode(ga, gb, ratio)
                        apprm = APPR(gm)
                    else:
                        gm = merge_two_graphs(ga, gb, data=False)
                        apprm = APPR(gm)
                    for seed in nodes_in_common:
                        if strategy in ["NWF", "WAPPR", "AllocationRW", "AllocationRW-ZB", "OneBased"]:
                            cm = apprm.compute_appr(seed)
                        # elif strategy == "allocation-rw":
                            # cm = apprm.compute_allocation_appr(
                            #     seed, ga, gb, c=ratio)
                        # elif strategy == "dynamic-rw":
                        #     cm = apprm.compute_dynamic_appr(
                        #         seed, ga, gb, c=ratio)
                        # elif strategy == "DynamicRW":
                        #     cm = apprm.compute_dynamic_weighting_appr(
                        #         seed, ga, gb, c=ratio)
                        # elif strategy == "DynamicRW-ZB":
                        #     cm = apprm.compute_dynamic_weighting_appr(
                        #         seed, ga, gb, c=ratio, one_based=False)
                        elif strategy == "ClassicRW":
                            cm = apprm.compute_aclcut_c_appr(seed, omega=ratio)
                        elif strategy == "RelaxedRW":
                            cm = apprm.compute_aclcut_r_appr(
                                seed, r=ratio)
                        elif strategy == "WAPPRS":
                            cm = apprm.compute_appr_with_supernode(seed)
                        elif strategy == "ML-LCD":
                            cm = mllcd.compute_mllcd(seed)
                            node_in_order = cm
                        cm.sort()
                        line = f'{seed}'
                        for nd in cm:
                            line += f' {nd}'
                        line += '\n'
                        f.write(line)

                        if strategy != "ML-LCD":
                            node_in_order = apprm.get_node_in_order()
                        line = f'{seed}'
                        for nd in node_in_order:
                            line += f' {nd}'
                        line += '\n'
                        f_top.write(line)

                        if strategy == "ML-LCD":
                            continue

                        conductances = apprm.get_cond_profile()
                        line = f'{seed}'
                        for cond in conductances:
                            line += f' {cond}'
                        line += '\n'
                        f_conductance.write(line)

                    f.close()
                    print(
                        "ETA:",
                        eta.eta(
                            (j + 1 + i * len(RATIOS)) /
                            (len(RATIOS) * len(STRATEGIES))
                        )
                    )
        # notify_slack(
        #     f"output_merged_clusters for ({ganame}, {gbname}) ended",
        #     f"#samples: {NUM_SAMPLES}, total_time: {eta.total_time()}"
        # )
    pass


def exp_true_removing_overlaps():
    PATHS = [
        # ("graph/Email-Enron.txt", "graph/CA-GrQc.txt"),
        # ("graph/0.1-0.01-3-100-normalorder.gr",
        #  "graph/0.3-0.01-3-100-mixedorder.gr"),
        # ("graph/socfb-Caltech36.mtx", "graph/web-edu.mtx"),
        # ("graph/web-edu.mtx", "graph/CA-GrQc.txt"),
        # ("graph/socfb-Caltech36.mtx", "graph/CA-GrQc.txt"),
        # ("graph/Email-Enron.txt", "graph/web-edu.mtx")
        # ("graph/aucs-lunch.gr", "graph/aucs-facebook.gr"),
        ("graph/Airports-Lufthansa.gr", "graph/Airports-Ryanair.gr"),
        # ("graph/dkpol-ff.gr", "graph/dkpol-Re.gr"),
        ("graph/Rattus-DI.gr",
         "graph/Rattus-PA.gr"),
    ]
    STRATEGIES = [
        "NWF",
        "ClassicRW",
        "RelaxedRW",
        "WAPPR",
        "WAPPRS"
        # "ML-LCD",
        # "AllocationRW",
        # "AllocationRW-ZB",
        # "DynamicRW",
        # "DynamicRW-ZB",
    ]
    NUM_PLOTS = 21
    RATIOS = [1 / LARGE] + [(i + 1) / (NUM_PLOTS + 1)
                            for i in range(NUM_PLOTS)] + [1 - 1 / LARGE]
    percents = [0.01, 0.05, 0.1, 0.25, 0.5, 1]
    labels = ["1%", "5%", "10%", "25%", "50%", "100%"]

    for ga_path, gb_path in PATHS:
        ganame = get_gname(ga_path)
        gbname = get_gname(gb_path)
        dash_index = ganame.find("-")
        dname = ganame[:dash_index]
        laname = ganame[dash_index + 1:]
        lbname = gbname[dash_index + 1:]
        cas = read_clusters(f"cluster/{ganame}/appr.txt")
        cbs = read_clusters(f"cluster/{gbname}/appr.txt")
        cas_mllcd = read_clusters(f"cluster/{ganame}/mllcd.txt")
        cbs_mllcd = read_clusters(f"cluster/{gbname}/mllcd.txt")
        filenames_tks_a_med = []
        for strategy in STRATEGIES:
            list_meds_tks_a, list_errs_tks_a = [], []
            list_meds_tks_b, list_errs_tks_b = [], []
            list_meds_tks_ab, list_errs_tks_ab = [], []
            for percent in percents:
                data_tks_a = {ratio: [] for ratio in RATIOS}
                data_tks_b = {ratio: [] for ratio in RATIOS}
                data_tks_ab = {ratio: [] for ratio in RATIOS}
                min_top_sizes = defaultdict(lambda: float('inf'))
                removing_percent = f"-removing-{percent}" if percent != 1 else ""
                for ratio in RATIOS:
                    cms = read_clusters(
                        f"cluster/{ganame}-{gbname}/{strategy}-{ratio}{removing_percent}.txt")
                    topks = read_clusters(
                        f"cluster/{ganame}-{gbname}-top/{strategy}-{ratio}{removing_percent}.txt",
                        return_type=list,
                    )
                    l_cms = [(k, v) for k, v in cms.items()]
                    l_cms.sort()
                    for seed, cm in l_cms:
                        top_size = len(topks[seed])
                        min_top_sizes[seed] = min(
                            min_top_sizes[seed], top_size)
                for ratio in RATIOS:
                    topks = read_clusters(
                        f"cluster/{ganame}-{gbname}-top/{strategy}-{ratio}{removing_percent}.txt",
                        return_type=list,
                    )
                    l_topks = [(k, v) for k, v in topks.items()]
                    l_topks.sort()
                    for i, (seed, topk) in enumerate(l_topks):
                        if strategy == "ML-LCD":
                            ca = cas_mllcd[seed]
                            cb = cbs_mllcd[seed]
                        else:
                            ca = cas[seed]
                            cb = cbs[seed]
                        ca_size = min(len(ca), min_top_sizes[seed])
                        cb_size = min(len(cb), min_top_sizes[seed])
                        top_a = set(topk[:ca_size])
                        top_b = set(topk[:cb_size])
                        tks_a = len(top_a.intersection(ca)) / len(top_a)
                        tks_b = len(top_b.intersection(cb)) / len(top_b)
                        data_tks_a[ratio].append(tks_a)
                        data_tks_b[ratio].append(tks_b)
                        if tks_a != 0:
                            data_tks_ab[ratio].append(tks_b / tks_a)
                        else:
                            data_tks_ab[ratio].append(100)
                df_tks_a = pd.DataFrame(data_tks_a).loc[:, RATIOS]
                df_tks_b = pd.DataFrame(data_tks_b).loc[:, RATIOS]
                df_tks_ab = pd.DataFrame(data_tks_ab).loc[:, RATIOS]
                meds_tks_a, errs_tks_a = convert_to_meds_and_errors_q1q3(
                    df_tks_a)
                meds_tks_b, errs_tks_b = convert_to_meds_and_errors_q1q3(
                    df_tks_b)
                meds_tks_ab, errs_tks_ab = convert_to_meds_and_errors_q1q3(
                    df_tks_ab)
                list_meds_tks_a.append(meds_tks_a)
                list_errs_tks_a.append(errs_tks_a)
                list_meds_tks_b.append(meds_tks_b)
                list_errs_tks_b.append(errs_tks_b)
                list_meds_tks_ab.append(meds_tks_ab)
                list_errs_tks_ab.append(errs_tks_ab)

            filename = f'tmp/med-tks-a-{ganame}-{gbname}-{strategy}.pdf'
            draw_chart(
                [RATIOS] * len(list_meds_tks_a),
                list_meds_tks_a,
                # list_errorbars=list_errs_tks_a,
                labels=labels,
                legend_title="overlap",
                title=f'{strategy}, {dname} (A: {laname}, B: {lbname})',
                x_axis_title="r",
                y_axis_title="TKS",
                loc="lower right",
                left=-0.01,
                right=1.01,
                top=1.01,
                bottom=-0.01,
                filename=filename,
                print_filename=True,
            )
            filenames_tks_a_med.append(filename)

            filename = f'tmp/med-tks-a-{gbname}-{ganame}.pdf'
            for i, l in enumerate(list_meds_tks_b):
                list_meds_tks_b[i] = l[::-1]
            for i, l in enumerate(list_errs_tks_b):
                list_errs_tks_b[i][0] = l[0][::-1]
                list_errs_tks_b[i][1] = l[1][::-1]
            draw_chart(
                [RATIOS] * len(list_meds_tks_b),
                list_meds_tks_b,
                labels=labels,
                title=f'{strategy}, {dname} (A: {lbname}, B: {laname})',
                x_axis_title="r",
                y_axis_title="TKS",
                loc="lower right",
                left=-0.01,
                right=1.01,
                top=1.01,
                bottom=-0.01,
                filename=filename,
                print_filename=True,
            )
            filenames_tks_a_med.append(filename)

        filename = concatanate_images(
            filenames_tks_a_med,
            f"tmp/med-tks-a",
            3,
            2,
            print_filename=True,
        )
        upload_to_imgbb(filename)


def compute_mllcds():
    PATHS = [
        "graph/aucs-lunch.gr",
        "graph/aucs-facebook.gr",
        "graph/Airports-Lufthansa.gr",
        "graph/Airports-Ryanair.gr",
        # "graph/dkpol-ff.gr",
        # "graph/dkpol-Re.gr",
        # "graph/Rattus-DI.gr",
        # "graph/Rattus-PA.gr",
    ]
    for path in PATHS:
        compute_mllcd(path)


def compute_mllcd(path: str):
    g = read_graph(path)
    graph_name = get_gname(path)

    try:
        dir = "cluster/" + graph_name
        print(dir)
        os.makedirs(dir)
    except FileExistsError:
        pass
    f = open("cluster/" + graph_name + "/mllcd.txt", "w")
    mllcd = MLLCD([g], beta=0)
    eta = ETA()
    percent = len(g) / 100
    percent_done = 1
    for cnt, seed in enumerate(g, 1):
        cluster = mllcd.compute_mllcd(seed)
        cluster.sort()
        line = f'{seed}'
        for nd in cluster:
            line += f' {nd}'
        line += '\n'
        f.write(line)

        if cnt / percent >= percent_done:
            print(
                f'Completed: {percent_done}%',
                f'ETA: {round(eta.eta(cnt / len(g)), 1)}s',
                sep=', '
            )
            percent_done += 1
    f.close()
    pass


def exp_num_unique_nodes():
    PATHS = [
        ("graph/aucs-lunch.gr", "graph/aucs-facebook.gr"),
        ("graph/Airports-Lufthansa.gr", "graph/Airports-Ryanair.gr"),
        ("graph/dkpol-ff.gr", "graph/dkpol-Re.gr"),
        ("graph/Rattus-DI.gr",
         "graph/Rattus-PA.gr"),
    ]
    records = []
    labels = ["Graph A", "Graph B", "unique in A", "unique in B", "common"]
    for ga_path, gb_path in PATHS:
        ganame = get_gname(ga_path)
        gbname = get_gname(gb_path)
        ga = read_graph(ga_path)
        gb = read_graph(gb_path)
        ndsa = set([nd for nd in ga])
        ndsb = set([nd for nd in gb])
        nodes_in_common = list(ndsa.intersection(ndsb))
        record = [
            f"{ganame} (n = {ga.number_of_nodes()}, m = {ga.number_of_edges()})",
            f"{gbname} (n = {gb.number_of_nodes()}, m = {gb.number_of_edges()})",
            len(ga) - len(nodes_in_common),
            len(gb) - len(nodes_in_common),
            len(nodes_in_common),
        ]
        records.append(record)
    export_table(records, labels)
    pass


def weighted_degree(g, nd):
    w = 0
    if g.__class__ == nx.Graph:
        for nbr in g.neighbors(nd):
            w += g[nd][nbr]['weight']
    elif g.__class__ == nx.DiGraph:
        for nbr in g.successors(nd):
            w += g[nd][nbr]['weight']
        for nbr in g.predecessors(nd):
            w += g[nbr][nd]['weight']
    return w


def see_detail_of_methods():
    ga_path, gb_path = "graph/Rattus-DI.gr", "graph/Rattus-PA.gr"
    # ga_path, gb_path = "graph/aucs-lunch.gr", "graph/aucs-facebook.gr"
    # ga_path, gb_path = "graph/ex_a.gr", "graph/ex_b.gr"
    ganame = get_gname(ga_path)
    gbname = get_gname(gb_path)
    ga = read_graph(ga_path)
    gb = read_graph(gb_path)
    ndsa = set([nd for nd in ga])
    ndsb = set([nd for nd in gb])
    nodes_in_common = list(ndsa.intersection(ndsb))
    uniquenodes_a = set([nd for nd in ga]) - set(nodes_in_common)
    uniquenodes_b = set([nd for nd in gb]) - set(nodes_in_common)
    ratio = 1024
    ratio = 1 / (ratio + 1)
    nga, ngb = normalize_edge_weight(ga, gb, ratio)
    gm1 = merge_two_graphs(nga, ngb, data=True)
    # print(gm1.edges().data())
    wsum1 = 0
    for u, v in gm1.edges():
        wsum1 += gm1.get_edge_data(u, v)['weight']
    wsum2 = 0
    gm2 = merge_two_graphs_with_supernode(ga, gb, ratio)
    for u, v in gm2.edges():
        wsum2 += gm2.get_edge_data(u, v)['weight']
    print(wsum1, wsum2)
    # for seed in nodes_in_common:
    #     apprm = APPR(gm2)
    #     apprm.compute_appr_with_supernode(seed)
    #     appr_vec = apprm.get_appr_vec()
    #     print(len(appr_vec), end=", ")
    #     # apprm = APPR(gm1)
    #     # apprm.compute_appr(seed)
    #     # appr_vec = apprm.get_appr_vec()
    #     # print(len(appr_vec), end=", ")
    #     # gm3 = convert_two_graphs_to_digraph(ga, gb, ratio)
    #     # apprm = DAPPR(gm3)
    #     # apprm.compute_appr(seed)
    #     # appr_vec = apprm.get_appr_vec()
    #     # print(len(appr_vec), end=", ")
    #     # gm4 = convert_two_graphs_to_digraph_one_based(ga, gb, ratio)
    #     # apprm = DAPPR(gm4)
    #     # apprm.compute_appr(seed)
    #     # appr_vec = apprm.get_appr_vec()
    #     # print(len(appr_vec), end=", ")
    #     print()


def see_change_with_r():
    ga_path, gb_path = "graph/Airports-Lufthansa.gr", "graph/Airports-Ryanair.gr"
    # ga_path, gb_path = "graph/Rattus-DI.gr", "graph/Rattus-PA.gr"
    NUM_PLOTS = 21
    RATIOS = [1 / LARGE] + [(i + 1) / (NUM_PLOTS + 1)
                            for i in range(NUM_PLOTS)] + [1 - 1 / LARGE]
    ganame = get_gname(ga_path)
    gbname = get_gname(gb_path)
    dash_index = ganame.find("-")
    ga = read_graph(ga_path)
    gb = read_graph(gb_path)
    STRATEGY = "WAPPRS"
    ndsa = set([nd for nd in ga])
    ndsb = set([nd for nd in gb])
    nodes_in_common = list(ndsa.intersection(ndsb))
    all_nodes = ndsa.union(ndsb)
    SEED = random.choice(nodes_in_common)
    ca = read_clusters(
        f"cluster/{ganame}-{gbname}/{STRATEGY}-{RATIOS[0]}.txt")[SEED]
    cb = read_clusters(
        f"cluster/{ganame}-{gbname}/{STRATEGY}-{RATIOS[-1]}.txt")[SEED]
    com_in_common = ca.intersection(cb)
    only_a = ca - cb
    only_b = cb - ca
    rest = all_nodes - ca - cb
    nds = list(only_a) + list(com_in_common) + list(only_b) + list(rest)
    l = []
    print(f"seed: {SEED}")
    for ratio in RATIOS:
        cms = read_clusters(
            f"cluster/{ganame}-{gbname}/{STRATEGY}-{ratio}.txt")
        cm = cms[SEED]
        node_in_community = [nd in cm for nd in nds]
        print(f"ratio: {ratio}, community: {cm}")
        l.append(node_in_community)
    np_l = np.array(l).transpose()
    a_rows = [i for i in range(len(only_a))]
    common_rows = [i for i in range(
        len(only_a), len(only_a) + len(com_in_common))]
    b_rows = [i for i in range(
        len(only_a) + len(com_in_common), len(only_a) + len(com_in_common) + len(only_b))]
    rest_rows = [i for i in range(
        len(only_a) + len(com_in_common) + len(only_b), len(nds))]
    a_matrix = np_l[a_rows, :]
    common_matrix = np_l[common_rows, :]
    b_matrix = np_l[b_rows, :]
    rest_matrix = np_l[rest_rows, :]
    row_sums = np.sum(a_matrix, axis=1)
    sorted_indices = np.argsort(row_sums)
    sorted_matrix = a_matrix[sorted_indices, :]
    res = sorted_matrix
    res = np.vstack([res, common_matrix])
    row_sums = np.sum(b_matrix, axis=1)
    sorted_indices = np.argsort(row_sums)[::-1]
    sorted_matrix = b_matrix[sorted_indices, :]
    res = np.vstack([res, sorted_matrix])
    row_sums = np.sum(rest_matrix, axis=1)
    sorted_indices = np.argsort(row_sums)[::-1]
    sorted_matrix = rest_matrix[sorted_indices, :]
    res = np.vstack([res, sorted_matrix])
    # print(res)
    fig = plt.figure(figsize=(5, 5))  # in inches
    plt.imshow(res,
               cmap="Greys",
               interpolation="none")
    plt.savefig("tmp/yeah.png")


def playing():
    from metis import networkx_to_metis
    g = read_graph("/home/thenter/tmp/ego-facebook.txt")
    # Convert the graph to a format that can be used by METIS
    G = networkx_to_metis(g)

    # partition the graph using METIS
    part = metis.part_graph(G, 5)[1]

    id2part = []
    # print the partition
    for i, nd in enumerate(g):
        id2part.append([nd, part[i]])
    export_to_simple_file(id2part)


if __name__ == "__main__":
    # compute_apprs()
    # compute_mllcds()
    # output_merged_clusters()
    # compare_two_ratios()
    # check_hidden_with_seed()
    # draw_subgraph_around_specific_node()
    # exp_rwer_among_strategies()
    # exp_true_among_strategies()
    # exp_nodes_in_out_of_topk()
    # find_non_motonotonic_increases()
    # strategy_check()
    # create_newinfo_figs()
    # compare_diff_datasets()
    # check_median_and_draw_box()
    # conductance_distribution()
    # conductance_distribution_merged()
    # create_dataset_removing_overlaps()
    # output_merged_clusters_removing_overlaps()
    # exp_true_removing_overlaps()
    # exp_num_unique_nodes()
    # see_detail_of_methods()
    # playing()
    see_change_with_r()
    pass
