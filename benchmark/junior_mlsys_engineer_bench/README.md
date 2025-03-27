



# Machine Learning Systems Example Questions

Here we showcase Curie's performance on some common MLSys questions. 

### Q1: How does reducing the number of sampling steps affect the inference time of a pre-trained diffusion model on MNIST? What is the relationship between them (linear or sub-linear)?

```bash
python3 -m curie.main -f benchmark/junior_mlsys_engineer_bench/q1_diffusion_step.txt --report
```

- Detailed question: `q1_diffusion_step.txt`
- **Estimated runtime**: 33 min (Model serving is time-consuming.)
- **Estimated cost**: $2.8 
- **Sample log file**: Available [here](/docs/example_logs/mlsys_diffusion_step_20250327.log)
- **Sample report file**: Available [here](/docs/example_logs/mlsys_diffusion_step_20250327.md)
