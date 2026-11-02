"""拡散モデルの、実行時の設定（実行ステップ数 EXEC、生成ステップ数）を変えて、動作の診断を比べる。"""
import sys, importlib.util
sys.path.insert(0, "src")
import sorting.visuomotor as vm
exec_n, samp = int(sys.argv[2]), int(sys.argv[3])
vm.EXEC = exec_n; vm.DIFF_SAMPLE = samp
spec = importlib.util.spec_from_file_location("d", "scripts/vm_diag_motion.py"); d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)
print(f"EXEC={exec_n} 生成ステップ={samp}", d.run(sys.argv[1], list(range(10000, 10016)), steps=300), flush=True)
