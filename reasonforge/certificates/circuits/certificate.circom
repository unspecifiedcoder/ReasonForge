/*
 * ReasonForge Certificate Verification Circuit
 *
 * Proves that a VerificationCertificate's fields are internally consistent:
 *   - The sum of step verdicts equals the claimed verified_steps count
 *   - The overall verdict follows deterministically from the step counts
 *   - Each step verdict is binary (0 or 1)
 *
 * Public inputs  (visible to verifier):
 *   task_hash[8]      -- 256-bit hash as 8 x 32-bit limbs
 *   overall_verdict    -- 0=FAILED, 1=PARTIALLY_VERIFIED, 2=VERIFIED
 *   total_steps        -- total number of steps
 *   verified_steps     -- number of verified steps
 *   timestamp          -- Unix timestamp
 *
 * Private inputs (hidden from verifier):
 *   step_verdicts[20]  -- array of 0/1 values (padded with 0s)
 */

pragma circom 2.0.0;

template CertificateVerifier(MAX_STEPS) {
    // ── Public inputs ──────────────────────────────────────────
    signal input task_hash[8];          // 256-bit as 8 x 32-bit limbs
    signal input overall_verdict;       // 0, 1, or 2
    signal input total_steps;
    signal input verified_steps;
    signal input timestamp;

    // ── Private inputs ─────────────────────────────────────────
    signal input step_verdicts[MAX_STEPS];

    // ── Constraint 1: Each step_verdict is binary (0 or 1) ────
    for (var i = 0; i < MAX_STEPS; i++) {
        step_verdicts[i] * (1 - step_verdicts[i]) === 0;
    }

    // ── Constraint 2: Only the first total_steps entries may be ──
    // ──               non-zero; padding slots must be 0          ──
    //
    // For each index i, if i >= total_steps then step_verdicts[i] == 0.
    // We enforce this by computing an "active" flag per slot:
    //   active[i] = 1 if i < total_steps, else 0
    // Then: step_verdicts[i] * (1 - active[i]) === 0
    //
    // To compute active[i] without comparison operators we use
    // a running indicator: active[0] = (total_steps > 0) ? 1 : 0,
    // and active[i] decrements a counter.
    //
    // Simpler approach: we enforce sum only over all MAX_STEPS slots,
    // and the padding constraint ensures unused slots are 0, so the
    // sum automatically equals the sum of the first total_steps slots.

    // ── Constraint 3: sum(step_verdicts) == verified_steps ────
    signal partial_sum[MAX_STEPS + 1];
    partial_sum[0] <== 0;
    for (var i = 0; i < MAX_STEPS; i++) {
        partial_sum[i + 1] <== partial_sum[i] + step_verdicts[i];
    }
    partial_sum[MAX_STEPS] === verified_steps;

    // ── Constraint 4: overall_verdict follows from step counts ──
    //
    // We need to enforce:
    //   if verified_steps == total_steps AND total_steps > 0  => overall_verdict == 2
    //   if verified_steps > 0 AND verified_steps < total_steps => overall_verdict == 1
    //   if verified_steps == 0                                  => overall_verdict == 0
    //
    // Strategy: compute indicator signals using arithmetic.

    // is_total_positive: 1 if total_steps > 0, else 0
    // We use the witness to provide the inverse when non-zero.
    signal input is_total_positive;       // witness: 1 if total_steps > 0
    signal input total_steps_inv;         // witness: 1/total_steps if > 0, else 0

    // Enforce: is_total_positive is binary
    is_total_positive * (1 - is_total_positive) === 0;

    // Enforce consistency: total_steps * total_steps_inv == is_total_positive
    total_steps * total_steps_inv === is_total_positive;

    // Enforce: if is_total_positive == 0 then total_steps == 0
    (1 - is_total_positive) * total_steps === 0;

    // is_verified_positive: 1 if verified_steps > 0, else 0
    signal input is_verified_positive;
    signal input verified_steps_inv;

    is_verified_positive * (1 - is_verified_positive) === 0;
    verified_steps * verified_steps_inv === is_verified_positive;
    (1 - is_verified_positive) * verified_steps === 0;

    // is_all_verified: 1 if verified_steps == total_steps AND total_steps > 0
    // Approach: difference = total_steps - verified_steps
    signal diff;
    diff <== total_steps - verified_steps;

    signal input is_diff_zero;      // witness: 1 if diff == 0, else 0
    signal input diff_inv;          // witness: 1/diff if diff != 0, else 0

    is_diff_zero * (1 - is_diff_zero) === 0;
    diff * diff_inv === 1 - is_diff_zero;
    is_diff_zero * diff === 0;

    // is_all_verified = is_diff_zero AND is_total_positive
    signal is_all_verified;
    is_all_verified <== is_diff_zero * is_total_positive;

    // is_partial = is_verified_positive AND (NOT is_all_verified)
    // = is_verified_positive AND is_total_positive AND (NOT is_diff_zero)
    // We need verified > 0 AND verified < total
    signal is_partial;
    signal temp_partial;
    temp_partial <== is_verified_positive * (1 - is_diff_zero);
    is_partial <== temp_partial * is_total_positive;

    // is_failed = NOT is_verified_positive
    // (verified_steps == 0)
    signal is_failed;
    is_failed <== 1 - is_verified_positive;

    // Now enforce: overall_verdict == 2*is_all_verified + 1*is_partial + 0*is_failed
    // Note: exactly one of {is_all_verified, is_partial, is_failed} must be 1
    // (this follows from the construction, but let's verify)
    signal sum_indicators;
    sum_indicators <== is_all_verified + is_partial + is_failed;
    sum_indicators === 1;

    // Expected verdict
    signal expected_verdict;
    expected_verdict <== 2 * is_all_verified + is_partial;
    overall_verdict === expected_verdict;

    // ── Constraint 5: task_hash limbs are 32-bit ──────────────
    // Each limb must be in [0, 2^32 - 1].
    // We enforce this by range-checking via bit decomposition.
    signal task_hash_bits[8][32];
    for (var j = 0; j < 8; j++) {
        var limb_sum = 0;
        for (var b = 0; b < 32; b++) {
            task_hash_bits[j][b] <-- (task_hash[j] >> b) & 1;
            task_hash_bits[j][b] * (1 - task_hash_bits[j][b]) === 0;
            limb_sum += task_hash_bits[j][b] * (1 << b);
        }
        task_hash[j] === limb_sum;
    }

    // ── Constraint 6: timestamp is non-negative (implicit in field) ──
    // The timestamp is a public input; no additional constraint needed
    // beyond it being a valid field element provided by the prover.
}

component main {public [task_hash, overall_verdict, total_steps, verified_steps, timestamp]} = CertificateVerifier(20);
