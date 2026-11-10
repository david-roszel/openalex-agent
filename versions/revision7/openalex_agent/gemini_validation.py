"""Gemini-specific accounting/profile over the existing approved-workload runner."""
from dataclasses import asdict
from decimal import Decimal
import json

from .corpus import atomic_json
from .development_pilot import SpendLedger
from .gemini import GEMINI_VERSION, GeminiADCPlanner

GEMINI_VALIDATION_VERSION='gemini-3.8-adc-development-validation-v2-tool-contract'


class GeminiSpendLedger(SpendLedger):
    def save(self):
        super().save()
        value=json.loads(self.path.read_text())
        value.update(protocol=GEMINI_VALIDATION_VERSION,input_usd_per_million='1.50',output_usd_per_million='7.50',
            price_basis='Gross Standard; reasoning included; no trial/promotion/cache discounts assumed',
            cloud_trial_balance_independently_verified=False)
        atomic_json(self.path,value)

    def settle(self,index,diagnostic):
        row=self.attempts[index]
        row.update(http_status=diagnostic.get('http_status'),error_code=diagnostic.get('error_code'),
                   provider_usage=diagnostic.get('provider_usage'))
        incoming,outgoing=diagnostic.get('prompt_tokens'),diagnostic.get('completion_tokens')
        if type(incoming) is int and 0<=incoming<=1050000 and type(outgoing) is int and 0<=outgoing<=4096:
            cost=(Decimal(incoming)*Decimal('1.5')+Decimal(outgoing)*Decimal('7.5'))/1000000
            row.update(usage='provider_reported',retained_usd=str(cost),prompt_tokens=incoming,completion_tokens=outgoing,
                reasoning_tokens=diagnostic.get('reasoning_tokens'),visible_output_tokens=diagnostic.get('visible_output_tokens'),
                tool_use_prompt_tokens=diagnostic.get('tool_use_prompt_tokens'))
        elif incoming is not None or outgoing is not None:
            self.stop_reason='provider_usage_outside_approved_limits'
        if diagnostic.get('http_status') in (401,403,404):self.stop_reason='pilot_provider_access_failure'
        self.save()


class GeminiValidationBackend:
    protocol=GEMINI_VALIDATION_VERSION
    ledger=GeminiSpendLedger

    @staticmethod
    def profile(settings):
        return {**asdict(settings),'transport':GEMINI_VERSION,'auth':'ADC','sdk_version':'3.0.0',
                'trial_credit_deduction_verified':False,'price_basis':'gross-before-promotional-or-trial-credits'}

    @staticmethod
    def guarded_factory(settings,ledger,*,transport=None):
        class GuardedGemini(GeminiADCPlanner):
            def _request(self,body):
                attempt=ledger.reserve()
                self.deadline=ledger.started+ledger.max_seconds
                try:return super()._request(body)
                finally:ledger.settle(attempt,self.last_call)
        return lambda:GuardedGemini(settings,executor=transport)
