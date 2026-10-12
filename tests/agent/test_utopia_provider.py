import pytest
from agent.knowledge_provider import KnowledgeRequest
from plugins.knowledge.utopia import UtopiaKnowledgeProvider


def test_read_only_tool_allowlist_and_dispatch(monkeypatch):
    provider=UtopiaKnowledgeProvider()
    calls=[]
    def rpc(method, params=None):
        calls.append((method,params))
        return {'tools':[{'name':n,'inputSchema':{'type':'object'}} for n in ['search_chunks','remember','get_document','search_docs']]}
    monkeypatch.setattr(provider,'_rpc',rpc)
    names=[t['function']['name'] for t in provider.get_knowledge_tool_schemas()]
    assert names==['utopia_search_chunks','utopia_get_document']
    with pytest.raises(ValueError):provider.handle_knowledge_tool_call('utopia_remember',{})
    assert len(calls)==1
    provider.handle_knowledge_tool_call('utopia_get_document',{'document_id':'d'})
    assert calls[-1]==('tools/call',{'name':'get_document','arguments':{'document_id':'d'}})


def test_evidence_budget_provenance_and_no_silent_fallback(monkeypatch):
    provider=UtopiaKnowledgeProvider({'mcp_url':'http://localhost/api/v1/kbs/k/mcp'})
    monkeypatch.setattr(provider,'_rpc',lambda *a:{'structuredContent':{'kb_id':'k','limit':6,'chunks':[
        {'document_id':'d','chunk_id':'c','filename':'original.md','text':'abcdef'},
        {'document_id':'d2','chunk_id':'c2','filename':'second.md','text':'more'}]}})
    result=provider.retrieve_knowledge(KnowledgeRequest(query='q',backend='utopia',top_k=2,evidence_char_budget=4))
    assert result.actual_backend=='utopia' and not result.degraded
    assert [e.content for e in result.evidence]==['abcd']
    assert result.citations[0].resource_id=='d'
    assert result.citations[0].uri=='utopia://k/documents/d'
    assert result.evidence[0].provenance['kb_id']=='k'
    monkeypatch.setattr(provider,'_rpc',lambda *a:{'content':[{'type':'text','text':'legacy result'}]})
    with pytest.raises(RuntimeError,match='incompatible'):provider.retrieve_knowledge(KnowledgeRequest(query='q'))
