
Shader "Lab11/Blur"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }

    SubShader
    {
        Cull Off
        ZWrite Off
        ZTest Always

        CGINCLUDE

        #include "UnityCG.cginc"

        sampler2D _MainTex;
        float4 _MainTex_TexelSize;

        static const float W[9] =
        {
            0.01898,
            0.05590,
            0.12092,
            0.19212,
            0.22417,
            0.19212,
            0.12092,
            0.05590,
            0.01898
        };

        fixed4 blur(float2 uv, float2 dir)
        {
            fixed4 sum = 0;

            for (int k = -4; k <= 4; k++)
            {
                sum += W[k + 4] *
                tex2D(_MainTex, uv + dir * k);
            }

            return sum;
        }

        ENDCG

        // Прохід 0: горизонтальне розмиття
        Pass
        {
            CGPROGRAM

            #pragma vertex vert_img
            #pragma fragment fragHorizontal

            fixed4 fragHorizontal(v2f_img i) : SV_Target
            {
                return blur(
                    i.uv,
                    float2(_MainTex_TexelSize.x, 0)
                );
            }

            ENDCG
        }

        // Прохід 1: вертикальне розмиття
        Pass
        {
            CGPROGRAM

            #pragma vertex vert_img
            #pragma fragment fragVertical

            fixed4 fragVertical(v2f_img i) : SV_Target
            {
                return blur(
                    i.uv,
                    float2(0, _MainTex_TexelSize.y)
                );
            }

            ENDCG
        }
    }
}
