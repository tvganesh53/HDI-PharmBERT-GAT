"""
pipeline_hdi.py â€” HDI Pipeline
Combines PharmBERT P8 (severity) + PharmFusion P8 (interaction type).
 
Output format matches what app_phase_g.py classify route expects:
{
    "outputs": [
        {
            "severity": {
                "top_label": str,
                "top_score": float,
                "all_scores": [{"label": str, "score": float}, ...]
            },
            "interaction_type": {
                "top_label": str,
                "top_score": float,
                "all_scores": [{"label": str, "score": float}, ...],
                "node_id_used": str | None,
                "gat_lookup": bool
            },
            "summary": str
        },
        ...
    ]
}
"""
import logging
from predictor_pharmbert import pharmbert_predictor
from predictor_fusion    import pharmfusion_predictor
 
log = logging.getLogger(__name__)
 
 
class HDIPipeline:
    model_name = "hdi-pipeline"
 
    def __init__(self):
        self._bert   = pharmbert_predictor
        self._fusion = pharmfusion_predictor
 
    def load(self):
        log.info("HDIPipeline: loading PharmBERT P8 â€¦")
        self._bert.load()
        log.info("HDIPipeline: loading PharmFusion P8 â€¦")
        self._fusion.load()
        log.info("HDIPipeline: both models ready âœ“")
 
    @property
    def is_loaded(self) -> bool:
        return self._bert.is_loaded and self._fusion.is_loaded
 
    def predict(self, texts: list, node_ids: list = None) -> dict:
        """
        Returns dict with key "outputs" â€” list of per-text result dicts.
        Format matches app_phase_g.py classify route expectations exactly.
        """
        if not self.is_loaded:
            self.load()
 
        if node_ids is None:
            node_ids = [""] * len(texts)
 
        bert_results   = self._bert.predict(texts)
        fusion_results = self._fusion.predict(texts, node_ids=node_ids)
 
        outputs = []
        for bert_r, fusion_r, node_id in zip(bert_results, fusion_results, node_ids):
 
            # â”€â”€ severity block â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            severity_all_scores = [
                {"label": label, "score": score}
                for label, score in bert_r["scores"].items()
            ]
            severity = {
                "top_label":  bert_r["top_label"],
                "top_score":  bert_r["top_score"],
                "all_scores": severity_all_scores,
            }
 
            # â”€â”€ interaction_type block â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            fusion_all_scores = [
                {"label": label, "score": score}
                for label, score in fusion_r["scores"].items()
            ]
            gat_lookup  = bool(node_id and node_id in (
                self._fusion._node_to_idx or {}
            ))
            interaction_type = {
                "top_label":    fusion_r["top_label"],
                "top_score":    fusion_r["top_score"],
                "all_scores":   fusion_all_scores,
                "node_id_used": node_id if node_id else None,
                "gat_lookup":   gat_lookup,
            }
 
            # â”€â”€ summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            summary = (
                f"{fusion_r['top_label']} interaction â€” "
                f"{bert_r['top_label']} "
                f"({round(bert_r['top_score'] * 100, 1)}%)"
            )
 
            outputs.append({
                "severity":         severity,
                "interaction_type": interaction_type,
                "summary":          summary,
            })
 
        return {"outputs": outputs}
 
 
# Singleton
hdi_pipeline = HDIPipeline()
 






