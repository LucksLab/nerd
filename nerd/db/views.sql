DROP VIEW IF EXISTS probe_tc_fits_view;

CREATE VIEW probe_tc_fits_view AS
SELECT
    r.id              AS fit_run_id,
    r.rg_id,
    r.nt_id,
    r.fit_kind,
    rg.rg_label,
    pr.temperature,
    pr.probe,
    pr.probe_conc,
    pr.rt_protocol,
    pr.buffer_id,
    pr.construct_id,
    mc.disp_name      AS construct_name,
    mb.name           AS buffer_name,
    mn.site           AS nt_site,
    mn.base           AS nt_base,
    k.param_numeric   AS kobs,
    lk.param_numeric  AS log_kobs,
    kd.param_numeric  AS log_kdeg,
    kobs_err.param_numeric AS log_kobs_err,
    kd_err.param_numeric   AS log_kdeg_err,
    diag_r2.param_numeric  AS r2,
    meta.param_text   AS metadata_json
FROM probe_tc_fit_runs r
JOIN probe_tc_fit_params k
     ON k.fit_run_id = r.id AND k.param_name = 'kobs'
LEFT JOIN probe_tc_fit_params lk
     ON lk.fit_run_id = r.id AND lk.param_name = 'log_kobs'
LEFT JOIN probe_tc_fit_params kd
     ON kd.fit_run_id = r.id AND kd.param_name = 'log_kdeg'
LEFT JOIN probe_tc_fit_params kobs_err
     ON kobs_err.fit_run_id = r.id AND kobs_err.param_name = 'log_kobs_err'
LEFT JOIN probe_tc_fit_params kd_err
     ON kd_err.fit_run_id = r.id AND kd_err.param_name = 'log_kdeg_err'
LEFT JOIN probe_tc_fit_params meta
     ON meta.fit_run_id = r.id AND meta.param_name = 'metadata'
LEFT JOIN probe_tc_fit_params diag_r2
     ON diag_r2.fit_run_id = r.id AND diag_r2.param_name = 'diag:r2'
JOIN (
    SELECT
        rg_id,
        MAX(temperature)         AS temperature,
        MAX(probe)               AS probe,
        MAX(probe_concentration) AS probe_conc,
        MAX(rt_protocol)         AS rt_protocol,
        MAX(buffer_id)           AS buffer_id,
        MAX(construct_id)        AS construct_id
    FROM probe_reactions
    GROUP BY rg_id
) pr ON pr.rg_id = r.rg_id
JOIN probe_reaction_groups rg ON rg.rg_id = r.rg_id
LEFT JOIN meta_constructs  mc ON mc.id = pr.construct_id
LEFT JOIN meta_buffers     mb ON mb.id = pr.buffer_id
LEFT JOIN meta_nucleotides mn ON mn.id = r.nt_id;

DROP VIEW IF EXISTS tempgrad_2statefit_params_view;

CREATE VIEW tempgrad_2statefit_params_view AS
SELECT
    a.param_numeric  AS a,
    a_err.param_numeric AS a_err,
    b.param_numeric  AS b,
    b_err.param_numeric AS b_err,
    c.param_numeric  AS c,
    c_err.param_numeric AS c_err,
    d.param_numeric  AS d,
    d_err.param_numeric AS d_err,
    f.param_numeric  AS f,
    f_err.param_numeric AS f_err,
    g.param_numeric  AS g,
    g_err.param_numeric AS g_err,
    r.nt_id          AS nt_id,
    mn.site          AS nt_site,
    mn.base          AS nt_base,
    diag_model.param_text AS "diag:model",
    mc.disp_name     AS construct_disp_name,
    mb.name          AS buffer_name
FROM tempgrad_fit_runs r
LEFT JOIN tempgrad_fit_params a
     ON a.fit_run_id = r.id AND a.param_name = 'a'
LEFT JOIN tempgrad_fit_params a_err
     ON a_err.fit_run_id = r.id AND a_err.param_name = 'a_err'
LEFT JOIN tempgrad_fit_params b
     ON b.fit_run_id = r.id AND b.param_name = 'b'
LEFT JOIN tempgrad_fit_params b_err
     ON b_err.fit_run_id = r.id AND b_err.param_name = 'b_err'
LEFT JOIN tempgrad_fit_params c
     ON c.fit_run_id = r.id AND c.param_name = 'c'
LEFT JOIN tempgrad_fit_params c_err
     ON c_err.fit_run_id = r.id AND c_err.param_name = 'c_err'
LEFT JOIN tempgrad_fit_params d
     ON d.fit_run_id = r.id AND d.param_name = 'd'
LEFT JOIN tempgrad_fit_params d_err
     ON d_err.fit_run_id = r.id AND d_err.param_name = 'd_err'
LEFT JOIN tempgrad_fit_params f
     ON f.fit_run_id = r.id AND f.param_name = 'f'
LEFT JOIN tempgrad_fit_params f_err
     ON f_err.fit_run_id = r.id AND f_err.param_name = 'f_err'
LEFT JOIN tempgrad_fit_params g
     ON g.fit_run_id = r.id AND g.param_name = 'g'
LEFT JOIN tempgrad_fit_params g_err
     ON g_err.fit_run_id = r.id AND g_err.param_name = 'g_err'
LEFT JOIN tempgrad_fit_params diag_model
     ON diag_model.fit_run_id = r.id AND diag_model.param_name = 'diag:model'
LEFT JOIN (
    SELECT
        tg_id,
        MAX(buffer_id)    AS buffer_id,
        MAX(construct_id) AS construct_id
    FROM probe_tempgrad_groups
    GROUP BY tg_id
) tg ON tg.tg_id = r.tg_id
LEFT JOIN meta_constructs mc ON mc.id = tg.construct_id
LEFT JOIN meta_buffers    mb ON mb.id = tg.buffer_id
LEFT JOIN meta_nucleotides mn ON mn.id = r.nt_id
WHERE r.fit_kind = 'two_state_melt'
  AND r.data_source = 'probe_tc';
