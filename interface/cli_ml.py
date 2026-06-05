import os
import sys

import click
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aalgoi.core.smart_solver import SmartSolver


@click.group()
def ml():
    """Machine Learning tasks via aalgoi CLI."""
    pass


@ml.command("train-word2vec")
@click.option("--corpus", type=click.Path(exists=True), required=True,
              help="Path to text corpus file (one sentence per line)")
@click.option("--domain", default="general",
              help="Domain: legal, medical, finance, etc.")
@click.option("--vector-size", default=100, type=int,
              help="Embedding dimension")
@click.option("--epochs", default=10, type=int, help="Training epochs")
@click.option("--output", type=click.Path(), help="Save model to file")
def train_word2vec(corpus, domain, vector_size, epochs, output):
    """Train Word2Vec embeddings on domain-specific corpus."""
    with open(corpus, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    if not sentences:
        click.echo("Error: Empty corpus file", err=True)
        sys.exit(1)

    solver = SmartSolver()
    data = {
        "corpus": sentences,
        "vector_size": vector_size,
        "epochs": epochs,
        "domain": domain,
    }
    question = f"Train Word2Vec on {domain} corpus with {vector_size} dimensions"
    result = solver.ask(question, data)

    if result.get("success"):
        model = result.get("result", {}).get("model")
        vocab_size = result.get("result", {}).get("vocabulary_size", 0)

        click.echo(click.style("SUCCESS: Trained Word2Vec", fg="green"))
        click.echo(f"  Vocabulary size: {vocab_size}")
        click.echo(f"  Vector size: {vector_size}")
        click.echo(f"  Domain: {domain}")

        if output and model:
            model.save(output)
            click.echo(f"  Model saved to: {output}")
    else:
        click.echo(
            click.style(f"FAILED: {result.get('error', 'Unknown error')}", fg="red")
        )
        sys.exit(1)


@ml.command("similar-words")
@click.option("--model", type=click.Path(exists=True), required=True,
              help="Path to trained Word2Vec model")
@click.option("--word", required=True, help="Input word")
@click.option("--topn", default=5, type=int, help="Number of similar words")
@click.option("--domain", default=None, help="Domain bias for results")
def similar_words(model, word, topn, domain):
    """Find semantically similar words using trained embeddings."""
    try:
        from gensim.models import Word2Vec as W2V
        w2v_model = W2V.load(model)
    except Exception as e:
        click.echo(f"Error loading model: {e}", err=True)
        sys.exit(1)

    solver = SmartSolver()
    data = {
        "model": w2v_model,
        "input_word": word,
        "topn": topn,
        "domain": domain or "general",
    }
    result = solver.ask(f"Find {topn} words similar to {word}", data)

    if result.get("success"):
        similar = result.get("result", {}).get("similar_words", [])

        if not similar:
            click.echo(f"'{word}' not found in vocabulary")
            sys.exit(1)

        click.echo(f"Words similar to '{word}':")
        for item in similar:
            click.echo(f"  {item['word']}: {item['similarity']:.3f}")
    else:
        click.echo(
            click.style(f"FAILED: {result.get('error', 'Unknown error')}", fg="red")
        )
        sys.exit(1)


@ml.command("visualize-embeddings")
@click.option("--model", type=click.Path(exists=True), required=True,
              help="Path to trained Word2Vec model")
@click.option("--words", multiple=True, required=True,
              help="Words to visualize (provide multiple --words flags)")
@click.option("--method", type=click.Choice(["pca", "tsne"]), default="pca",
              help="Dimensionality reduction method")
@click.option("--output", type=click.Path(), help="Save visualization image to file")
def visualize_embeddings(model, words, method, output):
    """Visualize word embeddings in 2D."""
    try:
        from gensim.models import Word2Vec as W2V
        w2v_model = W2V.load(model)
    except Exception as e:
        click.echo(f"Error loading model: {e}", err=True)
        sys.exit(1)

    embeddings = []
    valid_words = []
    for word in words:
        if word in w2v_model.wv:
            embeddings.append(w2v_model.wv[word])
            valid_words.append(word)
        else:
            click.echo(f"Warning: '{word}' not in vocabulary, skipping")

    if not embeddings:
        click.echo(click.style("FAILED: None of the specified words are in the vocabulary", fg="red"))
        sys.exit(1)

    solver = SmartSolver()
    data = {
        "embeddings": np.array(embeddings),
        "words": valid_words,
        "method": method,
    }
    result = solver.ask(f"Visualize {len(valid_words)} words using {method}", data)

    if result.get("success"):
        viz_data = result.get("result", {}).get("visualization_data", {})

        click.echo(click.style(f"SUCCESS: {method.upper()} visualization generated", fg="green"))
        click.echo(f"  Words: {valid_words}")
        click.echo(f"  Explained variance: {viz_data.get('explained_variance', 'N/A')}")

        if output:
            try:
                import matplotlib.pyplot as plt
                coords = np.array(viz_data["coordinates"])
                plt.figure(figsize=(10, 8))
                plt.scatter(coords[:, 0], coords[:, 1])
                for i, word in enumerate(valid_words):
                    plt.annotate(word, (coords[i, 0], coords[i, 1]))
                plt.title(f"Word Embeddings ({method.upper()})")
                plt.savefig(output)
                click.echo(f"  Saved to: {output}")
            except ImportError:
                click.echo("Warning: matplotlib not available, cannot save plot")
    else:
        click.echo(
            click.style(f"FAILED: {result.get('error', 'Unknown error')}", fg="red")
        )
        sys.exit(1)
