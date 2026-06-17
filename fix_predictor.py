with open('predictor_pharmbert.py', 'rb') as f:
    content = f.read()

old = b'results.append({\r\n                "label":      top_label,\r\n                "score":      round(top_score, 4),\r\n                "all_scores": all_scores,\r\n           })\r\n\r\n        return results[0] if isinstance(inputs, str) else results'

new = b'classifications = [\r\n                {"label": self.label_names[i], "score": round(scores[i], 4), "reasoning": ""}\r\n                for i in range(len(self.label_names))\r\n            ]\r\n            classifications.sort(key=lambda x: x["score"], reverse=True)\r\n            results.append({"classifications": classifications})\r\n        return {"outputs": results}'

if old in content:
    content = content.replace(old, new)
    with open('predictor_pharmbert.py', 'wb') as f:
        f.write(content)
    print('Fixed successfully')
else:
    print('Still not found')
