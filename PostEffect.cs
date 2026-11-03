
using UnityEngine;

[ExecuteInEditMode]
[RequireComponent(typeof(Camera))]
public class PostEffect : MonoBehaviour
{
    public Material material;

    void OnRenderImage(RenderTexture src, RenderTexture dst)
    {
        if (material == null)
        {
            Graphics.Blit(src, dst);
            return;
        }

        RenderTexture tmp = RenderTexture.GetTemporary(
            src.width,
            src.height,
            0,
            src.format
        );

        try
        {
            Graphics.Blit(src, tmp, material, 0);
            Graphics.Blit(tmp, dst, material, 1);
        }
        finally
        {
            RenderTexture.ReleaseTemporary(tmp);
        }
    }
}
