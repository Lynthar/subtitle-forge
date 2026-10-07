"""Shared fixtures."""

import functools
import json

import httpx
import pytest


class FakeServer:
    """Records each request with its JSON body; ``respond(request, body)`` gives the response."""

    def __init__(self, respond):
        self.respond = respond
        self.requests = []

    def __call__(self, request):
        body = json.loads(request.content) if request.content else None
        self.requests.append((request, body))
        return self.respond(request, body)


@pytest.fixture
def serve(monkeypatch):
    """``serve(respond)`` routes every httpx.Client built afterwards to a FakeServer."""

    def install(respond):
        server = FakeServer(respond)
        transport = httpx.MockTransport(server)
        monkeypatch.setattr(httpx, "Client", functools.partial(httpx.Client, transport=transport))
        return server

    return install
