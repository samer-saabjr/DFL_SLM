from sharding import make_client_splits, save_shard_manifests

K = 2
out_dir = f"out/sst2_base/shards_category_overlap_k{K}"
info = make_client_splits(
    task_name="sst2", 
    k=K, 
    mode="category_overlap",
    category_key="label",  # Partition by sentiment label (non-IID!)
    overlap_ratio=0.15,
    seed=123
)
save_shard_manifests(out_dir, info)
print("sizes:", info["size_per_client"])