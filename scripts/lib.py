"""Shared settings and helpers for the measuring scripts (see docs/reproduce.md for the variables)."""
import json, os, subprocess, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def _p(name, default):
    return os.path.expanduser(os.environ.get(name, default))


ROOT = _p("ROOT", "~/llama.cpp")                      # llama.cpp tree with the patches applied
BUILD = os.environ.get("BUILD", "build-cuda")         # build directory inside it
MODEL = _p("MODEL", "~/models/inkling-2-mini/Inkling-2-Mini-UD-IQ3_XXS.gguf")
DRAFT = _p("DRAFT", "~/models/inkling-2-mini/dspark-Inkling-2-Mini-Q8_0.gguf")
CUDA = _p("CUDA", "/usr/local/cuda")
DBX = os.environ.get("DBX", "")                       # container to run the binaries in; empty runs them on the host
WORK = _p("WORK", "~/.cache/inkling-accel")           # logs, KL baselines, profiles
PORT = int(os.environ.get("PORT", "8082"))            # port of the throw-away test servers
os.makedirs(WORK, exist_ok=True)

# the usual flags of the speculative setup measured in the README
SPEC = (f"-ctk q4_0 -ctv q4_0 -ctkd q4_0 -ctvd q4_0 --model-draft {DRAFT} --spec-type draft-dspark --spec-draft-ngl 99 --spec-draft-n-max 2")


def bin_(name, root=None, build=None):
    return f"{root or ROOT}/{build or BUILD}/bin/{name}"


def argv(cmd, env=None, root=None, build=None, host=False):
    """The argv that runs a shell command with the libraries of the build and the given environment, in the container if one is set (host=True: never)."""
    exports = f"export LD_LIBRARY_PATH={root or ROOT}/{build or BUILD}/bin:{CUDA}/lib64; " + "".join(f"export {k}={v}; " for k, v in (env or {}).items())
    inner = exports + cmd
    return ["distrobox", "enter", DBX, "--", "sh", "-c", inner] if DBX and not host else ["sh", "-c", inner]


def sh(cmd, env=None, root=None, timeout=3600):
    r = subprocess.run(argv(cmd, env, root), capture_output=True, text=True, timeout=timeout)
    return r.stdout + r.stderr


def _pids():
    return subprocess.run(["pgrep", "-f", f"[l]lama-server .*--port {PORT}"], capture_output=True, text=True).stdout.split()


class Server:
    """A throw-away llama-server on localhost:PORT with the given extra arguments and environment."""

    def __init__(self, args, env=None, root=None, build=None, host=False):
        self.args, self.env, self.root, self.build, self.host = args, env or {}, root, build, host
        self.log = os.path.join(WORK, "server.log")

    def __enter__(self):
        cmd = f"{bin_('llama-server', self.root, self.build)} -m {MODEL} -ngl 99 --jinja --host 127.0.0.1 --port {PORT} {self.args}"
        self.proc = subprocess.Popen(argv(cmd, self.env, self.root, self.build, self.host), stdout=open(self.log, "w"), stderr=subprocess.STDOUT)
        for _ in range(100):
            time.sleep(3)
            if "listening" in open(self.log).read():
                return self
            if self.proc.poll() is not None:
                break
        errors = [l.strip() for l in open(self.log) if " E " in l][:2]
        self.__exit__()
        raise RuntimeError("server did not start (is the GPU free?): " + " | ".join(errors))

    def __exit__(self, *exc):
        for pid in _pids():
            subprocess.run(["kill", pid])
        for _ in range(20):
            if not _pids():
                break
            time.sleep(1)
        time.sleep(3)


PROMPTS = [("code", "Write a Python implementation of an LRU cache class with get and put methods, with docstrings and type hints."),
           ("prose", "Tell me in detail about the history of the Roman Empire, starting with the founding of Rome."),
           ("json", "Generate a JSON array of 15 objects with fields id, name, email, city, age. Output only JSON."),
           ("sql", "Write a PostgreSQL schema for a blog with users, posts, comments and tags, with indexes and foreign keys."),
           ("refactor", "Refactor this into clean TypeScript with types: function f(a,b){var r=[];for(var i=0;i<a.length;i++){if(a[i]>b){r.push(a[i]*2)}}return r}"),
           ("docs", "Write a README section explaining how to install, configure and run a small command line tool called todo-cli."),
           ("math", "Explain step by step how to solve the quadratic equation 3x^2 - 7x + 2 = 0 and verify the roots."),
           ("rust", "Write a Rust function that parses a CSV line into a Vec<String>, handling quoted fields, with unit tests.")]
LONG = "This is a test paragraph about the history of computing. " * 400   # about 6k tokens


def chat(messages, max_tokens=256, **extra):
    """One chat completion with thinking off and greedy sampling; returns (message, timings)."""
    body = {"model": "x", "max_tokens": max_tokens, "temperature": 0, "messages": messages, "chat_template_kwargs": {"enable_thinking": False}, **extra}
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(r, timeout=900))
    return d["choices"][0]["message"], d["timings"]
