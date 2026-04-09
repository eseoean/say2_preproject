#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*
 * Feature Engineering Pipeline for Drug Response Prediction
 * Project: 20260408_pre_project_biso_myprotocol
 *
 * Steps:
 *   1. prepare_fe_inputs   — Bridge preprocessing (labels, sample features, drug features)
 *   2. build_features      — Main FE: merge sample+drug+labels, impute, filter, normalize
 *   3. build_pair_features — Advanced FE: pathway + chemistry + LINCS + target features
 *   4. upload_results      — Upload all outputs to S3
 */

// ── Processes ──────────────────────────────────────────────

process prepare_fe_inputs {
    tag "prepare_inputs_${params.run_id}"

    input:
    val ready

    output:
    path 'fe_inputs/', emit: fe_inputs_dir

    script:
    """
    mkdir -p fe_inputs
    python3 /workspace/nextflow/scripts/prepare_fe_inputs.py \
        --label-uri '${params.gdsc_ic50_uri}' \
        --drug-uri '${params.drug_features_uri}' \
        --sample-uri '${params.depmap_crispr_long_uri}' \
        --output-prefix fe_inputs \
        --run-id '${params.run_id}' \
        --binary-quantile ${params.binary_quantile}
    """
}

process build_features {
    tag "build_features_${params.run_id}"

    input:
    path fe_inputs_dir

    output:
    path 'features/', emit: features_dir

    script:
    """
    mkdir -p features
    python3 /workspace/nextflow/scripts/build_features.py \
        --sample-feature-uri '${fe_inputs_dir}/sample_features.parquet' \
        --drug-feature-uri '${fe_inputs_dir}/drug_features.parquet' \
        --label-uri '${fe_inputs_dir}/labels.parquet' \
        --out-features features/features.parquet \
        --out-labels features/labels.parquet \
        --out-manifest features/manifest.json \
        --run-id '${params.run_id}'
    """
}

process build_pair_features {
    tag "pair_features_${params.run_id}"

    input:
    path features_dir
    path fe_inputs_dir

    output:
    path 'pair_features/', emit: pair_features_dir

    script:
    """
    mkdir -p pair_features
    python3 /workspace/nextflow/scripts/build_pair_features_newfe_v2.py \
        --pairs-uri '${features_dir}/labels.parquet' \
        --sample-expression-uri '${fe_inputs_dir}/sample_features.parquet' \
        --drug-uri '${fe_inputs_dir}/drug_features.parquet' \
        --lincs-drug-signature-uri '${params.lincs_drug_sig_uri}' \
        --drug-target-uri '${params.drug_target_uri}' \
        --smiles-col smiles \
        --out-dir pair_features \
        --run-id '${params.run_id}'
    """
}

process upload_results {
    tag "upload_${params.run_id}"

    input:
    path features_dir
    path pair_features_dir

    script:
    """
    aws s3 cp --recursive '${features_dir}' \
        '${params.fe_output_dir}/${params.run_id}/features/'
    aws s3 cp --recursive '${pair_features_dir}' \
        '${params.fe_output_dir}/${params.run_id}/pair_features/'
    echo "Uploaded to ${params.fe_output_dir}/${params.run_id}/"
    """
}

// ── Workflow ────────────────────────────────────────────────

workflow {
    // Trigger channel
    start_ch = Channel.of('start')

    // Step 1: Prepare FE inputs (bridge tables)
    prepare_fe_inputs(start_ch)

    fe_inputs = prepare_fe_inputs.out.fe_inputs_dir

    // Step 2: Build base features (merge + impute + filter + normalize)
    build_features(fe_inputs)

    features = build_features.out.features_dir

    // Step 3: Build pair features (pathway + chemistry + LINCS + target)
    build_pair_features(features, fe_inputs)

    // Step 4: Upload all results to S3
    upload_results(features, build_pair_features.out.pair_features_dir)
}
