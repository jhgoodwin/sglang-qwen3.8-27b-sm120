#!/usr/bin/env python3
"""OpenAI-compatible smoke checks; runtime qualification remains out of scope."""
import argparse, base64, json, pathlib, time, urllib.error, urllib.request
PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
def call(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode(); req = urllib.request.Request(url, data=data, headers={"content-type":"application/json"})
    with urllib.request.urlopen(req, timeout=30) as response: raw, ctype, status = response.read(), response.headers.get("content-type", ""), response.status
    if "text/event-stream" in ctype or (payload and payload.get("stream")):
        events = [json.loads(x[5:].strip()) for x in raw.decode(errors="replace").splitlines() if x.startswith("data:") and x[5:].strip() != "[DONE]"]
        return status, {"events": events, "raw": raw.decode(errors="replace")}
    if not raw: return status, {}
    try: return status, json.loads(raw)
    except json.JSONDecodeError: return status, {"raw": raw.decode(errors="replace")}
def assert_chat(body, label):
    assert isinstance(body.get("choices"), list) and body["choices"], f"{label}: missing choices"; assert "message" in body["choices"][0], f"{label}: missing message"
def main():
    p=argparse.ArgumentParser(); p.add_argument("--base-url",default="http://127.0.0.1:11436"); p.add_argument("--output",type=pathlib.Path,default=pathlib.Path("bench/results/smoke")); p.add_argument("--allow-no-server",action="store_true"); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True); results=[]; checks={}
    try:
        for path in ("/health","/v1/models"):
            status,body=call(a.base_url+path); assert status==200,f"{path}: HTTP {status}"; results.append({"path":path,"status":status,"body":body})
        # Keep visible-output checks deterministic.  With a small token budget,
        # Qwen3.8 may spend the entire response on reasoning_content otherwise.
        base={"model":"Qwen/Qwen3.8-27B","messages":[{"role":"user","content":"Reply with exactly: smoke-ok"}],"temperature":0,"max_tokens":16,"reasoning_effort":"none"}
        status,body=call(a.base_url+"/v1/chat/completions",base); assert status==200; assert_chat(body,"text"); checks["text"]=True; results.append({"path":"text","status":status,"body":body})
        status,body=call(a.base_url+"/v1/chat/completions",{**base,"stream":True}); assert status==200 and body["events"]; assert any(e.get("choices",[{}])[0].get("delta",{}).get("content") for e in body["events"]); checks["streaming"]=True; results.append({"path":"stream","status":status,"body":body})
        tool={**base,"messages":[{"role":"user","content":"Call the smoke function."}],"tools":[{"type":"function","function":{"name":"smoke","parameters":{"type":"object","properties":{}}}}],"tool_choice":{"type":"function","function":{"name":"smoke"}}}; status,body=call(a.base_url+"/v1/chat/completions",tool); assert status==200; assert_chat(body,"tool"); assert body["choices"][0]["message"].get("tool_calls"); checks["tool_calls"]=True; results.append({"path":"tool","status":status,"body":body})
        reasoning={**base,"reasoning_effort":"low","max_tokens":32}; status,body=call(a.base_url+"/v1/chat/completions",reasoning); assert status==200; assert_chat(body,"reasoning"); reasoning_message=body["choices"][0]["message"]; assert reasoning_message.get("reasoning_content") or reasoning_message.get("content"), "reasoning: missing content and reasoning_content"; checks["reasoning_path"]=True; results.append({"path":"reasoning","status":status,"body":body})
        image={**base,"messages":[{"role":"user","content":[{"type":"text","text":"Describe this image."},{"type":"image_url","image_url":{"url":"data:image/png;base64,"+PNG}}]}]}; status,body=call(a.base_url+"/v1/chat/completions",image); assert status==200; assert_chat(body,"image"); checks["image"]=True; results.append({"path":"image","status":status,"body":body})
        try: status,info=call(a.base_url+"/get_server_info"); results.append({"path":"/get_server_info","status":status,"body":info}); checks["server_info"]=status==200
        except urllib.error.HTTPError as e: results.append({"path":"/get_server_info","status":e.code,"unsupported":True}); checks["server_info"]=False
    except (AssertionError,OSError,urllib.error.URLError) as e:
        if not a.allow_no_server: raise SystemExit(f"smoke failed: {e}")
        results.append({"error":str(e),"gated":True})
    (a.output/"responses.json").write_text(json.dumps(results,indent=2)+"\n"); (a.output/"metadata.json").write_text(json.dumps({"base_url":a.base_url,"utc":time.time(),"checks":checks,"identity":"see /get_server_info result; unsupported is recorded"},indent=2)+"\n"); print(f"smoke results: {a.output}")
if __name__ == "__main__": main()
