use friedrich::gaussian_process::GaussianProcess;

/// Helper: build a simple 1-D GP with known training data.
fn make_gp() -> GaussianProcess<friedrich::kernel::Gaussian, friedrich::prior::ConstantPrior>
{
    let inputs = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
    let outputs = vec![0.0, 1.0, 0.0, -1.0, 0.0];
    GaussianProcess::default(inputs, outputs)
}

#[test]
fn interpolation_and_low_variance_at_training_points()
{
    let gp = make_gp();

    let inputs = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
    let expected = vec![0.0, 1.0, 0.0, -1.0, 0.0];

    let means: Vec<f64> = gp.predict(&inputs);
    let vars: Vec<f64> = gp.predict_variance(&inputs);

    for (i, (mean, exp)) in means.iter().zip(expected.iter()).enumerate()
    {
        assert!((mean - exp).abs() < 0.2, "mean at training point {i}: expected ≈ {exp}, got {mean}");
    }
    for (i, v) in vars.iter().enumerate()
    {
        assert!(*v < 0.5, "variance at training point {i} should be small, got {v}");
    }
}

#[test]
fn uncertainty_grows_away_from_data()
{
    let gp = make_gp();

    // Near a training point
    let near = vec![2.01];
    let var_near: f64 = gp.predict_variance(&near);

    // Far from any training point
    let far = vec![10.0];
    let var_far: f64 = gp.predict_variance(&far);

    assert!(var_far > var_near,
            "variance far from data ({var_far}) should exceed variance near data ({var_near})");
}

#[test]
fn predict_mean_variance_matches_separate_calls()
{
    let gp = make_gp();

    let inputs = vec![vec![0.5], vec![1.5], vec![5.0]];

    let means: Vec<f64> = gp.predict(&inputs);
    let vars: Vec<f64> = gp.predict_variance(&inputs);
    let (means2, vars2): (Vec<f64>, Vec<f64>) = gp.predict_mean_variance(&inputs);

    for i in 0..inputs.len()
    {
        assert!((means[i] - means2[i]).abs() < 1e-10, "mean mismatch at {i}: {} vs {}", means[i], means2[i]);
        assert!((vars[i] - vars2[i]).abs() < 1e-10, "variance mismatch at {i}: {} vs {}", vars[i], vars2[i]);
    }
}

#[test]
fn covariance_matrix_properties()
{
    let gp = make_gp();

    let inputs = vec![vec![0.5], vec![1.5], vec![2.5], vec![3.5]];
    let cov = gp.predict_covariance(&inputs);
    let vars: Vec<f64> = gp.predict_variance(&inputs);

    let n = inputs.len();
    assert_eq!(cov.nrows(), n);
    assert_eq!(cov.ncols(), n);

    // Symmetry
    for i in 0..n
    {
        for j in 0..n
        {
            assert!((cov[(i, j)] - cov[(j, i)]).abs() < 1e-10,
                    "covariance not symmetric at ({i},{j}): {} vs {}",
                    cov[(i, j)],
                    cov[(j, i)]);
        }
    }

    // Diagonal matches predict_variance
    for i in 0..n
    {
        assert!((cov[(i, i)] - vars[i]).abs() < 1e-10,
                "diagonal {i} mismatch: cov={} var={}",
                cov[(i, i)],
                vars[i]);
    }
}

#[test]
fn adding_samples_moves_prediction()
{
    let mut gp = make_gp();

    let test_point = vec![5.0];
    let mean_before: f64 = gp.predict(&test_point);

    let new_input = vec![vec![5.0]];
    let new_output = vec![10.0];
    gp.add_samples(&new_input, &new_output);

    let mean_after: f64 = gp.predict(&test_point);

    // After observing y=10 at x=5, the prediction there should shift toward 10.
    assert!((mean_after - 10.0).abs() < (mean_before - 10.0).abs(),
            "prediction should move toward new observation: before={mean_before}, after={mean_after}");
}
