#include <arm_neon.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void matmul_mask_f32_neon(
	const float *proto,
	const float *coeffs,
	uint8_t *out,
	int boxes,
	int channels,
	int area
) {
	float *acc = (float *)aligned_alloc(16, (size_t)area * sizeof(float));
	if (acc == 0) {
		return;
	}

	for (int b = 0; b < boxes; b++) {
		memset(acc, 0, (size_t)area * sizeof(float));

		const float *box_coeffs = coeffs + b * channels;

		for (int c = 0; c < channels; c++) {
			float coeff = box_coeffs[c];
			if (coeff == 0.0f) {
				continue;
			}

			float32x4_t coeff_v = vdupq_n_f32(coeff);
			const float *proto_ch = proto + c * area;

			int i = 0;
			for (; i + 16 <= area; i += 16) {
				float32x4_t acc0 = vld1q_f32(acc + i + 0);
				float32x4_t acc1 = vld1q_f32(acc + i + 4);
				float32x4_t acc2 = vld1q_f32(acc + i + 8);
				float32x4_t acc3 = vld1q_f32(acc + i + 12);

				float32x4_t p0 = vld1q_f32(proto_ch + i + 0);
				float32x4_t p1 = vld1q_f32(proto_ch + i + 4);
				float32x4_t p2 = vld1q_f32(proto_ch + i + 8);
				float32x4_t p3 = vld1q_f32(proto_ch + i + 12);

				acc0 = vfmaq_f32(acc0, p0, coeff_v);
				acc1 = vfmaq_f32(acc1, p1, coeff_v);
				acc2 = vfmaq_f32(acc2, p2, coeff_v);
				acc3 = vfmaq_f32(acc3, p3, coeff_v);

				vst1q_f32(acc + i + 0, acc0);
				vst1q_f32(acc + i + 4, acc1);
				vst1q_f32(acc + i + 8, acc2);
				vst1q_f32(acc + i + 12, acc3);
			}

			for (; i < area; i++) {
				acc[i] += proto_ch[i] * coeff;
			}
		}

		uint8_t *dst = out + b * area;

		int i = 0;
		uint8x16_t maskValue = vdupq_n_u8(4);

		for (; i + 16 <= area; i += 16) {
			float32x4_t a0 = vld1q_f32(acc + i + 0);
			float32x4_t a1 = vld1q_f32(acc + i + 4);
			float32x4_t a2 = vld1q_f32(acc + i + 8);
			float32x4_t a3 = vld1q_f32(acc + i + 12);

			uint32x4_t m0 = vcgtq_f32(a0, vdupq_n_f32(0.0f));
			uint32x4_t m1 = vcgtq_f32(a1, vdupq_n_f32(0.0f));
			uint32x4_t m2 = vcgtq_f32(a2, vdupq_n_f32(0.0f));
			uint32x4_t m3 = vcgtq_f32(a3, vdupq_n_f32(0.0f));

			uint16x4_t n0 = vmovn_u32(m0);
			uint16x4_t n1 = vmovn_u32(m1);
			uint16x4_t n2 = vmovn_u32(m2);
			uint16x4_t n3 = vmovn_u32(m3);

			uint16x8_t n01 = vcombine_u16(n0, n1);
			uint16x8_t n23 = vcombine_u16(n2, n3);
			uint8x16_t packed = vmovn_high_u16(vmovn_u16(n01), n23);

			packed = vandq_u8(packed, maskValue);
			vst1q_u8(dst + i, packed);
		}

		for (; i < area; i++) {
			dst[i] = acc[i] > 0.0f ? 4 : 0;
		}
	}

	free(acc);
}
