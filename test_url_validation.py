#!/usr/bin/env python3
"""
Script para validar que los URLs de recursos educativos funcionan correctamente
"""

import sys
import logging
from youtube_resources import YouTubeResourceFinder

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_url_validation():
    """Probar la validación de URLs"""

    finder = YouTubeResourceFinder()

    print("\n" + "="*70)
    print("PRUEBA DE VALIDACIÓN DE URLs - RECURSOS EDUCATIVOS")
    print("="*70 + "\n")

    # URLs de prueba: algunos válidos, algunos inválidos
    test_urls = [
        # URLs VÁLIDOS (dominios confiables)
        ("https://www.youtube.com/watch?v=test", "YouTube video"),
        ("https://www.khanacademy.org/math/algebra", "Khan Academy - Álgebra"),
        ("https://en.wikipedia.org/wiki/Calculus", "Wikipedia - Cálculo"),
        ("https://stackoverflow.com/questions/", "Stack Overflow"),
        ("https://www.brilliant.org/wiki/", "Brilliant.org"),
        ("https://www.codewars.com/kata/", "CodeWars"),
        ("https://github.com/topic/", "GitHub"),
        ("https://ocw.mit.edu/courses/", "MIT OpenCourseWare"),

        # URLs INVÁLIDOS (dominios no confiables o inexistentes)
        ("https://fake-educational-site.com/math", "Sitio falso"),
        ("https://malicious-domain.xyz/resources", "Dominio malicioso"),
        ("http://invalid.test/course", "Dominio inválido"),
        ("not-a-url", "Formato inválido"),
        ("", "URL vacío"),
    ]

    print("VALIDANDO URLs INDIVIDUALES:\n")

    valid_count = 0
    invalid_count = 0

    for url, description in test_urls:
        is_valid = finder.validate_url(url)
        status = "[OK]    " if is_valid else "[FAIL]  "

        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1

        print(f"{status} | {description:40} | {url}")

    print("\n" + "-"*70)
    print(f"Resultado: {valid_count} válidos, {invalid_count} inválidos\n")

    # Prueba de filtrado de recursos
    print("="*70)
    print("PRUEBA DE FILTRADO DE RECURSOS")
    print("="*70 + "\n")

    test_resources = [
        {
            'title': 'Khan Academy - Calculus',
            'url': 'https://www.khanacademy.org/math/calculus',
            'source': 'Khan Academy',
            'type': 'video'
        },
        {
            'title': 'Fake Resource',
            'url': 'https://fake-educational-site.com/resource',
            'source': 'Fake Site',
            'type': 'course'
        },
        {
            'title': 'Wikipedia - Integral',
            'url': 'https://en.wikipedia.org/wiki/Integral',
            'source': 'Wikipedia',
            'type': 'article'
        },
        {
            'title': 'Malicious Site',
            'url': 'https://malicious-domain.xyz/course',
            'source': 'Bad Site',
            'type': 'video'
        },
        {
            'title': 'GitHub - Math Projects',
            'url': 'https://github.com/topics/mathematics',
            'source': 'GitHub',
            'type': 'code'
        },
    ]

    print(f"Recursos ANTES de validación: {len(test_resources)}\n")

    valid_resources = finder.validate_and_filter_resources(test_resources)

    print(f"\nRecursos DESPUÉS de validación: {len(valid_resources)}\n")
    print("Recursos válidos:")
    for resource in valid_resources:
        print(f"  [OK] {resource['title']} ({resource['source']})")
        print(f"       {resource['url']}\n")

    if len(test_resources) > len(valid_resources):
        print(f"\nRecursos descartados: {len(test_resources) - len(valid_resources)}")

    # Prueba de búsqueda real
    print("\n" + "="*70)
    print("PRUEBA DE BÚSQUEDA DE RECURSOS PARA CÁLCULO")
    print("="*70 + "\n")

    resources = finder.search_educational_resources_multiformat(
        subject="Cálculo",
        risk_level="HIGH",
        language="es"
    )

    for category, items in resources.items():
        print(f"\n{category.upper()} ({len(items)} recursos):")
        for item in items[:2]:  # Mostrar solo los primeros 2
            print(f"  • {item.get('title', 'Sin título')}")
            print(f"    {item.get('url', 'Sin URL')}")

    total = sum(len(items) for items in resources.values())
    print(f"\nTotal de recursos: {total}")
    print("\n" + "="*70)
    print("[DONE] PRUEBA COMPLETADA")
    print("="*70 + "\n")

if __name__ == '__main__':
    try:
        test_url_validation()
    except Exception as e:
        logger.error(f"Error en prueba: {str(e)}", exc_info=True)
        sys.exit(1)
