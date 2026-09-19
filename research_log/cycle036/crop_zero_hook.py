"""Evaluation-only insertion after actual pooled crop assignment; production unchanged."""
import ast
import hashlib
import inspect
import textwrap
import types
import torch


def zero_crop(c):
    return torch.zeros_like(c)


def transformed_source(source):
    tree = ast.parse(textwrap.dedent(source))
    matches = []
    class Insert(ast.NodeTransformer):
        def visit_Assign(self, node):
            if any(isinstance(t, ast.Name) and t.id == 'crop_features' for t in node.targets):
                assert ast.unparse(node.value) == 'self.encode_images(ref_images).mean(dim=1)'
                matches.append(node)
                return [node, ast.parse('crop_features = _cycle036_zero_crop(crop_features)').body[0]]
            return node
    tree = Insert().visit(tree)
    assert len(matches) == 1
    return ast.unparse(ast.fix_missing_locations(tree))


def install(model):
    original = model.build_ref_input_embeddings.__func__
    source = textwrap.dedent(inspect.getsource(original))
    changed = transformed_source(source)
    calls = []
    def counted(c):
        calls.append({'shape': list(c.shape), 'dtype': str(c.dtype)})
        return zero_crop(c)
    namespace = dict(original.__globals__, _cycle036_zero_crop=counted)
    exec(compile(changed, '<cycle036-crop-zero>', 'exec'), namespace)
    model.build_ref_input_embeddings = types.MethodType(namespace[original.__name__], model)
    return {'original_method_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'intervened_method_sha256': hashlib.sha256(changed.encode()).hexdigest(),
            'location': 'LISA.py build_ref_input_embeddings immediately after pooled crop_features assignment',
            'calls': calls}


def self_check():
    for dtype in [torch.float32, torch.bfloat16]:
        c = torch.tensor([[1., -2., 3.]], dtype=dtype)
        b = torch.tensor([[4., 5., -6.]], dtype=dtype)
        saved_c, saved_b = c.clone(), b.clone()
        out = zero_crop(c) + b
        assert torch.equal(out, b) and torch.equal(b, saved_b) and torch.equal(c, saved_c)
        assert out.dtype == dtype and out.shape == c.shape and out.device == c.device
    example = 'def f(self, ref_images, bbox_features):\n    crop_features = self.encode_images(ref_images).mean(dim=1)\n    return crop_features + bbox_features\n'
    namespace = {'_cycle036_zero_crop': zero_crop}
    exec(transformed_source(example), namespace)
    dummy = types.SimpleNamespace(encode_images=lambda _: torch.ones(1, 2, 3))
    bbox = torch.tensor([[4., 5., -6.]])
    assert torch.equal(namespace['f'](dummy, None, bbox), bbox)
    print('PASS: pooled crop removed; bbox/input tensors, dtype, shape and device preserved')


if __name__ == '__main__':
    self_check()
