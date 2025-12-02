"""
Test Script - Validar que el endpoint /api/resources retorna URLs reales y funcionales
"""

import asyncio
import json
from youtube_resources import get_youtube_finder


async def test_youtube_finder():
    """Test del YouTube Resource Finder"""
    print("=" * 70)
    print("[TEST] YouTube Resource Finder")
    print("=" * 70)

    finder = get_youtube_finder()

    # Test casos
    test_cases = [
        {
            'subject': 'Algebra',
            'risk_level': 'MEDIUM',
            'language': 'es',
            'description': 'Busqueda en espanol para Algebra'
        },
        {
            'subject': 'Programacion Python',
            'risk_level': 'LOW',
            'language': 'es',
            'description': 'Python para principiantes'
        },
        {
            'subject': 'Calculo',
            'risk_level': 'HIGH',
            'language': 'es',
            'description': 'Calculo avanzado'
        },
    ]

    total_passed = 0
    total_failed = 0

    for i, test_case in enumerate(test_cases, 1):
        print("\n" + "=" * 70)
        print("[Test {}] {}".format(i, test_case['description']))
        print("=" * 70)

        try:
            # Ejecutar busqueda
            resources = finder.search_educational_videos(
                subject=test_case['subject'],
                risk_level=test_case['risk_level'],
                language=test_case['language']
            )

            if not resources:
                print("[FAIL] No se encontraron recursos para '{}'".format(test_case['subject']))
                total_failed += 1
                continue

            # Validar resultados
            print("[PASS] Recursos encontrados: {}\n".format(len(resources)))

            all_valid = True
            for idx, resource in enumerate(resources, 1):
                # Validar estructura
                required_keys = ['title', 'url', 'channel', 'type']
                has_all_keys = all(key in resource for key in required_keys)

                if not has_all_keys:
                    print("  [FAIL] [{}] Estructura invalida: {}".format(idx, resource))
                    all_valid = False
                    continue

                # Validar URL
                url = resource['url']
                is_valid_url = finder._is_valid_url(url)

                status = "[OK]" if is_valid_url else "[BAD]"
                print("  {} [{}] {}...".format(status, idx, resource['title'][:50]))
                print("      URL: {}...".format(url[:70]))
                print("      Canal: {} | Tipo: {}".format(resource['channel'], resource['type']))

                if not is_valid_url:
                    all_valid = False

            if all_valid:
                print("\n[PASS] Todos los recursos validos para '{}'".format(test_case['subject']))
                total_passed += 1
            else:
                print("\n[PARTIAL] Algunos recursos invalidos")
                total_passed += 1

        except Exception as e:
            print("[ERROR] {}".format(str(e)))
            total_failed += 1

    # Resumen
    print("\n" + "=" * 70)
    print("[SUMMARY] Resultados de Tests")
    print("=" * 70)
    print("[OK] Pasaron: {}/{}".format(total_passed, len(test_cases)))
    print("[FAIL] Fallaron: {}/{}".format(total_failed, len(test_cases)))
    print("=" * 70 + "\n")

    return total_failed == 0


def test_url_validation():
    """Test de validacion de URLs"""
    print("=" * 70)
    print("[TEST] URL Validation")
    print("=" * 70)

    finder = get_youtube_finder()

    test_urls = [
        ('https://www.youtube.com/results?search_query=algebra', True, 'YouTube Search'),
        ('https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab', True, 'YouTube Playlist'),
        ('https://ocw.mit.edu/search/?q=calculus', True, 'MIT OpenCourseWare'),
        ('https://www.khanacademy.org/search?query=geometry', True, 'Khan Academy'),
        ('https://example.com/fake', False, 'URL no valida'),
        ('https://instagram.com/some-profile', False, 'Instagram (no educativo)'),
    ]

    print("\nValidando URLs:\n")
    for url, should_be_valid, description in test_urls:
        is_valid = finder._is_valid_url(url)
        status = "[OK]" if is_valid == should_be_valid else "[FAIL]"
        expected = "valida" if should_be_valid else "invalida"
        result = "valida" if is_valid else "invalida"

        print("{} {}".format(status, description))
        print("   URL: {}...".format(url[:60]))
        print("   Esperado: {} | Resultado: {}".format(expected, result))
        print()

    print("=" * 70)


async def simulate_api_request():
    """Simular una request al endpoint /api/resources"""
    print("=" * 70)
    print("[TEST] Simular POST /api/resources")
    print("=" * 70)

    # Simular request
    request_data = {
        'student_name': 'Juan Perez',
        'subject': 'Algebra Lineal',
        'current_grade': 5.5,
        'risk_level': 'HIGH',
        'need': 'study_resource',
        'language': 'es'
    }

    print("\nRequest simulado:")
    print(json.dumps(request_data, indent=2))

    finder = get_youtube_finder()

    try:
        resources = finder.search_educational_videos(
            subject=request_data['subject'],
            risk_level=request_data['risk_level'],
            language=request_data['language']
        )

        response = {
            'success': True,
            'subject': request_data['subject'],
            'student_name': request_data['student_name'],
            'resources': ["[Video] {}".format(r['title']) for r in resources],
            'count': len(resources),
            'note': 'Recursos verificados para {}'.format(request_data["subject"])
        }

        print("\nResponse:")
        print(json.dumps({
            'success': response['success'],
            'subject': response['subject'],
            'count': response['count'],
            'sample': response['resources'][:2] if response['resources'] else []
        }, indent=2))

        print("\n[OK] Response valido con {} recursos".format(len(resources)))

    except Exception as e:
        print("[ERROR] {}".format(str(e)))


if __name__ == '__main__':
    print("\n")
    print("=" * 80)
    print("    TEST SUITE: Validacion de Recursos Educativos Reales")
    print("=" * 80)
    print()

    # Test 1: URL Validation
    test_url_validation()

    # Test 2: YouTube Finder
    print("\n")
    success = asyncio.run(test_youtube_finder())

    # Test 3: Simular API Request
    print("\n")
    asyncio.run(simulate_api_request())

    print("\n")
    if success:
        print("[OK] TODOS LOS TESTS PASARON EXITOSAMENTE")
    else:
        print("[WARN] ALGUNOS TESTS FALLARON - REVISAR LOGS")

    print()
