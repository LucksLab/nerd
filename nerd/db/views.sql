DROP VIEW IF EXISTS probe_tc_arrhenius_view;

CREATE VIEW probe_tc_arrhenius_view AS
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
