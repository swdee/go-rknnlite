#include <arm_neon.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// matmul_mask_f32_neon performs the segmentation prototype matrix
// multiplication used by YOLO segmentation models.
//
// For each detected object:
//
//   output_mask = sum(proto[channel] * coeff[channel])
//
// The accumulated float mask is then thresholded:
//
//   > 0.0f -> 4
//   <= 0.0f -> 0
//
// The output values intentionally match the original Go implementation
// (0 or 4) so that later OpenCV interpolation produces visually identical
// mask edges during resize operations.
//
// The implementation uses ARM NEON SIMD instructions to accelerate both:
//
//   1. float accumulation
//   2. binary threshold packing
void matmul_mask_f32_neon(
	const float *proto,
	const float *coeffs,
	uint8_t *out,
	int boxes,
	int channels,
	int area
) {
    // temporary accumulation buffer reused for each object mask
	float *acc = (float *)aligned_alloc(16, (size_t)area * sizeof(float));
	if (acc == 0) {
		return;
	}

    // process each detected object independently
	for (int b = 0; b < boxes; b++) {
	    // clear accumulation buffer for current object
		memset(acc, 0, (size_t)area * sizeof(float));

        // coefficients for this object
		const float *box_coeffs = coeffs + b * channels;

        // accumulate all prototype channels
		for (int c = 0; c < channels; c++) {
			float coeff = box_coeffs[c];

			// skip empty coefficients to reduce unnecessary work
			if (coeff == 0.0f) {
				continue;
			}

            // broadcast coefficient across NEON vector
			float32x4_t coeff_v = vdupq_n_f32(coeff);

			// pointer to prototype channel data
			const float *proto_ch = proto + c * area;

			int i = 0;

			// SIMD accumulate 16 pixels per iteration
			for (; i + 16 <= area; i += 16) {
			    // load accumulation buffer
				float32x4_t acc0 = vld1q_f32(acc + i + 0);
				float32x4_t acc1 = vld1q_f32(acc + i + 4);
				float32x4_t acc2 = vld1q_f32(acc + i + 8);
				float32x4_t acc3 = vld1q_f32(acc + i + 12);

                // load prototype values
				float32x4_t p0 = vld1q_f32(proto_ch + i + 0);
				float32x4_t p1 = vld1q_f32(proto_ch + i + 4);
				float32x4_t p2 = vld1q_f32(proto_ch + i + 8);
				float32x4_t p3 = vld1q_f32(proto_ch + i + 12);

				// fused multiply-add:
				// acc += proto * coeff
				acc0 = vfmaq_f32(acc0, p0, coeff_v);
				acc1 = vfmaq_f32(acc1, p1, coeff_v);
				acc2 = vfmaq_f32(acc2, p2, coeff_v);
				acc3 = vfmaq_f32(acc3, p3, coeff_v);

                // store updated accumulation
				vst1q_f32(acc + i + 0, acc0);
				vst1q_f32(acc + i + 4, acc1);
				vst1q_f32(acc + i + 8, acc2);
				vst1q_f32(acc + i + 12, acc3);
			}

            // scalar tail for remaining pixels
			for (; i < area; i++) {
				acc[i] += proto_ch[i] * coeff;
			}
		}

        // output mask for current object
		uint8_t *dst = out + b * area;

		int i = 0;

		// object pixels use value 4 to match original Go implementation
		uint8x16_t maskValue = vdupq_n_u8(4);

        // threshold accumulated floats into binary mask
		for (; i + 16 <= area; i += 16) {
		    // load accumulated floats
			float32x4_t a0 = vld1q_f32(acc + i + 0);
			float32x4_t a1 = vld1q_f32(acc + i + 4);
			float32x4_t a2 = vld1q_f32(acc + i + 8);
			float32x4_t a3 = vld1q_f32(acc + i + 12);

            // compare against zero
			uint32x4_t m0 = vcgtq_f32(a0, vdupq_n_f32(0.0f));
			uint32x4_t m1 = vcgtq_f32(a1, vdupq_n_f32(0.0f));
			uint32x4_t m2 = vcgtq_f32(a2, vdupq_n_f32(0.0f));
			uint32x4_t m3 = vcgtq_f32(a3, vdupq_n_f32(0.0f));

            // pack comparison masks down to bytes
			uint16x4_t n0 = vmovn_u32(m0);
			uint16x4_t n1 = vmovn_u32(m1);
			uint16x4_t n2 = vmovn_u32(m2);
			uint16x4_t n3 = vmovn_u32(m3);

			uint16x8_t n01 = vcombine_u16(n0, n1);
			uint16x8_t n23 = vcombine_u16(n2, n3);
			uint8x16_t packed = vmovn_high_u16(vmovn_u16(n01), n23);

            // convert 0xFF mask values into 4
			packed = vandq_u8(packed, maskValue);

			// store binary output mask
			vst1q_u8(dst + i, packed);
		}

        // scalar tail thresholding
		for (; i < area; i++) {
			dst[i] = acc[i] > 0.0f ? 4 : 0;
		}
	}

	free(acc);
}
