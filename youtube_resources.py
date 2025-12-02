"""
Módulo para buscar recursos educativos en YouTube
Sin requerir API key - usa web scraping responsable
"""

import logging
from typing import List, Dict, Any
from urllib.parse import urlencode, quote
import re

logger = logging.getLogger(__name__)


class YouTubeResourceFinder:
    """Encuentra recursos educativos en YouTube de forma confiable"""

    # Palabras clave educativas para mejorar búsquedas
    ACADEMIC_KEYWORDS = {
        'matemáticas': ['cálculo', 'álgebra', 'geometría', 'trigonometría', 'estadística'],
        'algebra': ['polinomios', 'ecuaciones', 'factorización', 'gráficas'],
        'geometría': ['triángulos', 'círculos', 'volumen', 'área'],
        'cálculo': ['derivadas', 'integrales', 'límites', 'funciones'],
        'física': ['mecánica', 'termodinámica', 'óptica', 'electricidad'],
        'química': ['reacciones', 'elementos', 'moléculas', 'estequiometría'],
        'biología': ['células', 'evolución', 'genética', 'ecosistemas'],
        'historia': ['antigua', 'medieval', 'moderna', 'contemporánea'],
        'lengua': ['gramática', 'literatura', 'redacción', 'ortografía'],
        'english': ['grammar', 'vocabulary', 'pronunciation', 'listening'],
        'programación': ['python', 'javascript', 'estructuras', 'algoritmos'],
        'programacion': ['python', 'javascript', 'estructuras', 'algoritmos'],
    }

    # Plataformas educativas confiables
    TRUSTED_CHANNELS = {
        'Khan Academy': 'https://www.youtube.com/@khanacademy',
        '3Blue1Brown': 'https://www.youtube.com/@3blue1brown',
        'Crash Course': 'https://www.youtube.com/@crashcourse',
        'TED-Ed': 'https://www.youtube.com/@teded',
        'MIT OpenCourseWare': 'https://www.youtube.com/@mitocw',
        'Stanford Online': 'https://www.youtube.com/@stanfordonline',
        'Professor Leonard': 'https://www.youtube.com/@ProfessorLeonard',
        'PatrickJMT': 'https://www.youtube.com/@patrickjmt',
    }

    # Dominios educativos CONFIABLES verificados
    TRUSTED_DOMAINS = {
        'khanacademy.org': True,
        'youtube.com': True,
        'youtu.be': True,
        'wikipedia.org': True,
        'brilliant.org': True,
        'codewars.com': True,
        'leetcode.com': True,
        'stackoverflow.com': True,
        'github.com': True,
        'medium.com': True,
        'dev.to': True,
        'reddit.com': True,
        'discord.com': True,
        'ocw.mit.edu': True,
        'coursera.org': True,
        'edx.org': True,
        'desmos.com': True,
        'geogebra.org': True,
        'wolframalpha.com': True,
        'phet.colorado.edu': True,
        'python.org': True,
        'docs.python.org': True,
        'developer.mozilla.org': True,
        'mdn.io': True,
        'preply.com': True,
        'superprof.com': True,
    }

    def __init__(self):
        """Inicializar el finder"""
        logger.info("YouTubeResourceFinder inicializado")
        self._url_cache = {}  # Caché de URLs validadas

    def search_educational_resources_multiformat(
        self,
        subject: str,
        risk_level: str = "MEDIUM",
        language: str = "es"
    ) -> Dict[str, List[Dict]]:
        """
        Buscar MÚLTIPLES FORMATOS de recursos educativos

        Retorna: Videos, Artículos, Ejercicios, Apps Interactivas, Documentación, Comunidades

        Args:
            subject: Tema a buscar (ej: 'Álgebra', 'Programación')
            risk_level: Nivel de dificultad (LOW, MEDIUM, HIGH)
            language: Idioma (es=español, en=inglés)

        Returns:
            Dict con categorías: videos, articles, exercises, interactive, docs, communities
        """
        try:
            logger.info(f"Buscando MÚLTIPLES FORMATOS para: {subject}")

            resources_by_format = {
                'videos': self._get_video_resources(subject, risk_level, language),
                'articles': self._get_article_resources(subject, language),
                'exercises': self._get_exercise_resources(subject, risk_level),
                'interactive': self._get_interactive_resources(subject),
                'documentation': self._get_documentation_resources(subject),
                'communities': self._get_community_resources(subject, language),
            }

            total_before = sum(len(v) for v in resources_by_format.values())
            logger.info(f"Total recursos antes de validación: {total_before}")

            # VALIDAR: Filtrar recursos con URLs inválidas
            validated_resources = {}
            for category, resources in resources_by_format.items():
                validated_resources[category] = self.validate_and_filter_resources(resources)

            total_after = sum(len(v) for v in validated_resources.values())
            logger.info(f"✓ {total_after} recursos válidos en {len(validated_resources)} categorías para {subject}")
            if total_before > total_after:
                logger.warning(f"⚠️ {total_before - total_after} recursos descartados por URLs inválidas")

            return validated_resources

        except Exception as e:
            logger.error(f"Error buscando recursos multi-formato: {str(e)}")
            return self._get_fallback_resources_multiformat(subject)

    def search_educational_videos(
        self,
        subject: str,
        risk_level: str = "MEDIUM",
        language: str = "es"
    ) -> List[Dict[str, str]]:
        """
        DEPRECATED: Usar search_educational_resources_multiformat() en su lugar
        Buscar solo videos educativos en YouTube

        Args:
            subject: Tema a buscar (ej: 'Álgebra', 'Programación')
            risk_level: Nivel de dificultad (LOW, MEDIUM, HIGH)
            language: Idioma (es=español, en=inglés)

        Returns:
            Lista de recursos con title, url, channel
        """
        try:
            logger.info(f"Buscando videos educativos para: {subject}")

            # Mejorar búsqueda con palabras clave académicas
            search_query = self._enhance_search_query(subject, risk_level, language)

            # Construir URLs de búsqueda directas (sin API)
            resources = self._get_direct_youtube_urls(search_query, subject)

            # Filtrar solo recursos confiables/educativos
            filtered = self._filter_educational_content(resources)

            logger.info(f"✓ {len(filtered)} videos encontrados para {subject}")
            return filtered

        except Exception as e:
            logger.error(f"Error buscando videos: {str(e)}")
            return self._get_fallback_resources(subject)

    def _enhance_search_query(self, subject: str, risk_level: str, language: str) -> str:
        """Mejorar query de búsqueda con palabras clave académicas"""
        subject_lower = subject.lower()

        # Buscar palabras clave relacionadas
        extra_keywords = ""
        for key, keywords in self.ACADEMIC_KEYWORDS.items():
            if key in subject_lower:
                extra_keywords = keywords[0]
                break

        # Construir query mejorada
        if language == "es":
            difficulty = {
                "LOW": "fácil principiante",
                "MEDIUM": "tutorial",
                "HIGH": "avanzado"
            }.get(risk_level, "tutorial")
            query = f"{subject} {difficulty} educativo tutorial"
        else:
            difficulty = {
                "LOW": "beginner easy",
                "MEDIUM": "tutorial",
                "HIGH": "advanced"
            }.get(risk_level, "tutorial")
            query = f"{subject} {difficulty} educational tutorial"

        if extra_keywords:
            query += f" {extra_keywords}"

        return query

    def _get_direct_youtube_urls(self, search_query: str, subject: str) -> List[Dict[str, str]]:
        """
        Obtener URLs de YouTube de forma directa (sin API key)
        Retorna URLs de búsqueda seguras
        """
        try:
            # Usar búsqueda web en lugar de librería deprecada
            resources = []

            # Método 1: URLs directas de búsqueda (100% funcional)
            search_url = f"https://www.youtube.com/results?search_query={quote(search_query)}"

            # Método 2: Canales educativos conocidos por tema
            educational_videos = self._get_trusted_channel_videos(subject)
            resources.extend(educational_videos)

            # Método 3: URLs de playlists conocidas por tema
            playlist_urls = self._get_educational_playlists(subject)
            resources.extend(playlist_urls)

            logger.info(f"URLs directas obtenidas: {len(resources)}")
            return resources

        except Exception as e:
            logger.error(f"Error en búsqueda directa: {str(e)}")
            return []

    def _get_trusted_channel_videos(self, subject: str) -> List[Dict[str, str]]:
        """Retornar videos de canales educativos confiables"""
        resources = []
        subject_lower = subject.lower()

        # Khan Academy - cobertura amplia
        if any(x in subject_lower for x in ['álgebra', 'geometría', 'cálculo', 'matemáticas', 'fisica', 'química']):
            resources.append({
                'title': 'Khan Academy - ' + subject,
                'url': 'https://www.youtube.com/results?search_query=khan+academy+' + quote(subject),
                'channel': 'Khan Academy',
                'type': 'video'
            })

        # 3Blue1Brown para matemáticas avanzadas
        if any(x in subject_lower for x in ['álgebra', 'cálculo', 'lineal', 'matemáticas']):
            resources.append({
                'title': '3Blue1Brown - ' + subject,
                'url': 'https://www.youtube.com/@3blue1brown/search?query=' + quote(subject),
                'channel': '3Blue1Brown',
                'type': 'video'
            })

        # Crash Course para historia, ciencias
        if any(x in subject_lower for x in ['historia', 'biología', 'química', 'física']):
            resources.append({
                'title': 'Crash Course - ' + subject,
                'url': 'https://www.youtube.com/@crashcourse/search?query=' + quote(subject),
                'channel': 'Crash Course',
                'type': 'video'
            })

        # MIT OpenCourseWare
        resources.append({
            'title': 'MIT OpenCourseWare - ' + subject,
            'url': f'https://ocw.mit.edu/search/?q={quote(subject)}',
            'channel': 'MIT OpenCourseWare',
            'type': 'video'
        })

        return resources

    def _get_educational_playlists(self, subject: str) -> List[Dict[str, str]]:
        """Retornar playlists educativas conocidas por tema"""
        resources = []
        subject_lower = subject.lower()

        playlists = {
            'álgebra': [
                ('3Blue1Brown - Essence of Algebra', 'https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab'),
                ('Khan Academy - Algebra 1', 'https://www.youtube.com/playlist?list=PL0G2sPzkx24o1QuGWbq4FH2F3awJ_n1Kb'),
            ],
            'geometría': [
                ('3Blue1Brown - Essence of Geometry', 'https://www.youtube.com/playlist?list=PLZHQObOWTQDfZ2L2mnqV7RZvXEQOyqW_E'),
            ],
            'cálculo': [
                ('3Blue1Brown - Essence of Calculus', 'https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr28EL8nNLT93aHOHeGrw'),
                ('Professor Leonard - Calculus', 'https://www.youtube.com/playlist?list=PLF0b3ThojznSMD3V02Ez0lVW3bnxGCZcA'),
            ],
            'programación': [
                ('freeCodeCamp - Python', 'https://www.youtube.com/results?search_query=freecodecamp+python'),
                ('Programming with Mosh', 'https://www.youtube.com/@programmingwithmosh/search?query=' + quote(subject)),
            ],
        }

        for keyword, playlist_list in playlists.items():
            if keyword in subject_lower:
                for title, url in playlist_list:
                    resources.append({
                        'title': title,
                        'url': url,
                        'channel': title.split(' - ')[0],
                        'type': 'playlist'
                    })

        return resources

    def _filter_educational_content(self, resources: List[Dict]) -> List[Dict]:
        """Filtrar solo contenido educativo confiable"""
        filtered = []

        for resource in resources:
            # Validar que la URL sea válida
            if self._is_valid_url(resource.get('url', '')):
                filtered.append(resource)

        return filtered[:10]  # Máximo 10 resultados

    def _is_valid_url(self, url: str) -> bool:
        """Validar que la URL sea de YouTube o plataforma educativa"""
        valid_domains = [
            'youtube.com',
            'youtu.be',
            'ocw.mit.edu',
            'coursera.org',
            'edx.org',
            'khanacademy.org',
        ]

        return any(domain in url.lower() for domain in valid_domains)

    def _get_video_resources(self, subject: str, risk_level: str, language: str) -> List[Dict]:
        """Obtener recursos de VIDEO"""
        videos = self._get_trusted_channel_videos(subject)
        videos.extend(self._get_educational_playlists(subject))
        return videos[:5]  # Top 5

    def _get_article_resources(self, subject: str, language: str) -> List[Dict]:
        """Obtener recursos de ARTÍCULOS y tutoriales escritos"""
        articles = [
            {
                'title': f'Wikipedia - {subject}',
                'url': f'https://es.wikipedia.org/wiki/{quote(subject)}' if language == 'es' else f'https://en.wikipedia.org/wiki/{quote(subject)}',
                'source': 'Wikipedia',
                'type': 'article',
                'description': 'Artículo enciclopédico con definiciones y conceptos'
            },
            {
                'title': f'Medium - Artículos sobre {subject}',
                'url': f'https://medium.com/search?q={quote(subject)}',
                'source': 'Medium',
                'type': 'article',
                'description': 'Artículos de expertos en blogs'
            },
            {
                'title': f'Dev.to - {subject} (Programación)',
                'url': f'https://dev.to/search?q={quote(subject)}',
                'source': 'Dev.to',
                'type': 'article',
                'description': 'Comunidad de desarrolladores con tutoriales'
            },
            {
                'title': f'Documentación oficial - {subject}',
                'url': f'https://www.google.com/search?q={quote(subject)}+documentación+oficial',
                'source': 'Docs',
                'type': 'article',
                'description': 'Documentación técnica oficial'
            },
        ]
        return articles[:4]

    def _get_exercise_resources(self, subject: str, risk_level: str) -> List[Dict]:
        """Obtener recursos de EJERCICIOS PRÁCTICOS"""
        difficulty = {
            'LOW': 'fácil',
            'MEDIUM': 'intermedio',
            'HIGH': 'avanzado'
        }.get(risk_level, 'intermedio')

        exercises = [
            {
                'title': f'Khan Academy - Ejercicios {subject}',
                'url': f'https://www.khanacademy.org/search?page_search_query={quote(subject)}',
                'source': 'Khan Academy',
                'type': 'exercise',
                'description': f'Problemas prácticos con soluciones paso a paso'
            },
            {
                'title': f'Brilliant.org - {subject}',
                'url': f'https://brilliant.org/courses/?=&page=1&sort=popular',
                'source': 'Brilliant',
                'type': 'exercise',
                'description': 'Problemas interactivos y desafiantes'
            },
            {
                'title': f'CodeWars - Ejercicios de Programación',
                'url': f'https://www.codewars.com/kata/search/{quote(subject)}',
                'source': 'CodeWars',
                'type': 'exercise',
                'description': f'Katas de programación nivel {difficulty}'
            },
            {
                'title': f'LeetCode - Problemas algorítmicos',
                'url': f'https://leetcode.com/problemset/all/?search={quote(subject)}',
                'source': 'LeetCode',
                'type': 'exercise',
                'description': 'Problemas de entrevista técnica'
            },
        ]
        return exercises[:4]

    def _get_interactive_resources(self, subject: str) -> List[Dict]:
        """Obtener recursos INTERACTIVOS (apps, simuladores)"""
        interactive = [
            {
                'title': f'Desmos - Calculadora gráfica',
                'url': 'https://www.desmos.com/calculator',
                'source': 'Desmos',
                'type': 'interactive',
                'description': 'Grafica funciones y experimenta en tiempo real'
            },
            {
                'title': f'GeoGebra - Geometría interactiva',
                'url': 'https://www.geogebra.org/classic',
                'source': 'GeoGebra',
                'type': 'interactive',
                'description': 'Construcciones geométricas dinámicas'
            },
            {
                'title': f'Wolfram Alpha - Motor computacional',
                'url': f'https://www.wolframalpha.com/input/?i={quote(subject)}',
                'source': 'Wolfram',
                'type': 'interactive',
                'description': 'Resuelve y visualiza problemas matemáticos'
            },
            {
                'title': f'PhET - Simulaciones científicas',
                'url': 'https://phet.colorado.edu/es/simulations/filter?subjects=&types=html,prototype',
                'source': 'PhET',
                'type': 'interactive',
                'description': 'Simulaciones interactivas de física y química'
            },
        ]
        return interactive[:4]

    def _get_documentation_resources(self, subject: str) -> List[Dict]:
        """Obtener DOCUMENTACIÓN oficial y guías"""
        docs = [
            {
                'title': f'Stack Overflow - {subject}',
                'url': f'https://stackoverflow.com/search?q={quote(subject)}',
                'source': 'Stack Overflow',
                'type': 'documentation',
                'description': 'Q&A de problemas específicos resueltos'
            },
            {
                'title': f'MDN Web Docs - (Web)',
                'url': f'https://developer.mozilla.org/es/search?q={quote(subject)}',
                'source': 'MDN',
                'type': 'documentation',
                'description': 'Referencia técnica de Web Development'
            },
            {
                'title': f'Python Docs - (Programación)',
                'url': 'https://docs.python.org/3/',
                'source': 'Python',
                'type': 'documentation',
                'description': 'Documentación oficial de Python'
            },
            {
                'title': f'Guías de estudio - {subject}',
                'url': f'https://www.google.com/search?q={quote(subject)}+guía+estudio+pdf',
                'source': 'Study Guides',
                'type': 'documentation',
                'description': 'Guías compiladas de estudiantes'
            },
        ]
        return docs[:4]

    def _get_community_resources(self, subject: str, language: str) -> List[Dict]:
        """Obtener COMUNIDADES y tutorías"""
        communities = [
            {
                'title': f'Reddit - r/{subject.replace(" ", "")}',
                'url': f'https://www.reddit.com/search/?q={quote(subject)}',
                'source': 'Reddit',
                'type': 'community',
                'description': 'Comunidad de estudiantes compartiendo dudas y soluciones'
            },
            {
                'title': f'Discord - Servidores educativos',
                'url': 'https://discord.gg/',
                'source': 'Discord',
                'type': 'community',
                'description': 'Comunidades en tiempo real para estudiar juntos'
            },
            {
                'title': f'GitHub - {subject} (Proyectos)',
                'url': f'https://github.com/search?q={quote(subject)}',
                'source': 'GitHub',
                'type': 'community',
                'description': 'Proyectos y código de ejemplo'
            },
            {
                'title': f'Tutorías - Preply/Superprof',
                'url': f'https://www.preply.com/es/tutores/',
                'source': 'Tutorías',
                'type': 'community',
                'description': 'Conecta con tutores particulares en línea'
            },
        ]
        return communities[:4]

    def _get_fallback_resources_multiformat(self, subject: str) -> Dict[str, List[Dict]]:
        """Fallback para multi-formato cuando falla todo"""
        return {
            'videos': [
                {
                    'title': f'Khan Academy - {subject}',
                    'url': f'https://www.khanacademy.org/search?page_search_query={quote(subject)}',
                    'source': 'Khan Academy',
                    'type': 'video'
                },
                {
                    'title': f'YouTube - {subject} educativo',
                    'url': f'https://www.youtube.com/results?search_query={quote(subject)}+educativo+tutorial',
                    'source': 'YouTube',
                    'type': 'video'
                },
            ],
            'articles': [
                {
                    'title': f'Wikipedia - {subject}',
                    'url': f'https://es.wikipedia.org/wiki/{quote(subject)}',
                    'source': 'Wikipedia',
                    'type': 'article'
                },
            ],
            'exercises': [
                {
                    'title': f'Khan Academy Ejercicios',
                    'url': f'https://www.khanacademy.org/search?page_search_query={quote(subject)}',
                    'source': 'Khan Academy',
                    'type': 'exercise'
                },
            ],
            'interactive': [
                {
                    'title': f'Desmos',
                    'url': 'https://www.desmos.com/calculator',
                    'source': 'Desmos',
                    'type': 'interactive'
                },
            ],
            'documentation': [
                {
                    'title': f'Stack Overflow',
                    'url': f'https://stackoverflow.com/search?q={quote(subject)}',
                    'source': 'Stack Overflow',
                    'type': 'documentation'
                },
            ],
            'communities': [
                {
                    'title': f'Reddit - {subject}',
                    'url': f'https://www.reddit.com/search/?q={quote(subject)}',
                    'source': 'Reddit',
                    'type': 'community'
                },
            ]
        }

    def _get_fallback_resources(self, subject: str) -> List[Dict[str, str]]:
        """Recursos fallback cuando falla la búsqueda (DEPRECATED)"""
        logger.warning(f"Usando fallback para {subject}")

        return [
            {
                'title': 'Khan Academy - ' + subject,
                'url': f'https://www.khanacademy.org/search?page_search_query={quote(subject)}',
                'channel': 'Khan Academy',
                'type': 'video'
            },
            {
                'title': 'YouTube Search - ' + subject,
                'url': f'https://www.youtube.com/results?search_query={quote(subject)}+educativo+tutorial',
                'channel': 'YouTube',
                'type': 'search'
            },
            {
                'title': 'MIT OpenCourseWare - ' + subject,
                'url': f'https://ocw.mit.edu/search/?q={quote(subject)}',
                'channel': 'MIT',
                'type': 'course'
            }
        ]

    def validate_url(self, url: str) -> bool:
        """
        Validar que una URL sea confiable y accesible

        Estrategia de 3 niveles:
        1. Verificar que pertenece a dominio confiable (TRUSTED_DOMAINS)
        2. Verificar estructura básica de URL
        3. HTTP HEAD request rápido (timeout 2s)
        """
        if not url:
            return False

        # Check cache
        if url in self._url_cache:
            return self._url_cache[url]

        try:
            # Nivel 1: Validar estructura básica
            if not url.startswith(('http://', 'https://')):
                self._url_cache[url] = False
                return False

            # Nivel 2: Extraer dominio
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Remover www. para comparación
            domain_check = domain.replace('www.', '')

            # Nivel 3: Verificar contra whitelist de dominios confiables
            is_trusted = any(
                domain_check == trusted or domain_check.endswith('.' + trusted)
                for trusted in self.TRUSTED_DOMAINS.keys()
            )

            if not is_trusted:
                logger.warning(f"Dominio no confiable: {domain} - {url}")
                self._url_cache[url] = False
                return False

            # Nivel 4: HTTP HEAD request rápido
            try:
                import requests
                response = requests.head(
                    url,
                    timeout=2,
                    allow_redirects=True,
                    headers={'User-Agent': 'Mozilla/5.0 (Educational Resource Validator)'}
                )
                is_valid = response.status_code < 400
                self._url_cache[url] = is_valid

                if not is_valid:
                    logger.warning(f"URL retorna {response.status_code}: {url}")
                else:
                    logger.debug(f"✓ URL válida: {url}")

                return is_valid
            except requests.Timeout:
                # Timeout = asume válido si paso validación de dominio
                logger.info(f"Timeout validando URL (pero dominio es confiable): {url}")
                self._url_cache[url] = True
                return True
            except Exception as e:
                logger.warning(f"Error HTTP validando: {url} ({str(e)})")
                # Si dominio es confiable pero hay error HTTP, asumir válido
                self._url_cache[url] = True
                return True

        except Exception as e:
            logger.error(f"Error validando URL: {url} ({str(e)})")
            self._url_cache[url] = False
            return False

    def validate_and_filter_resources(self, resources: List[Dict]) -> List[Dict]:
        """
        Validar todos los recursos y filtrar URLs inválidos

        Retorna solo recursos con URLs válidos
        """
        valid_resources = []

        for resource in resources:
            url = resource.get('url', '')

            # Validar URL
            if self.validate_url(url):
                valid_resources.append(resource)
            else:
                logger.warning(f"Recurso descartado por URL inválida: {resource.get('title')}")

        logger.info(f"✓ {len(valid_resources)}/{len(resources)} recursos pasan validación")
        return valid_resources


# Instancia global
youtube_finder = None


def get_youtube_finder() -> YouTubeResourceFinder:
    """Obtener instancia del finder"""
    global youtube_finder
    if youtube_finder is None:
        youtube_finder = YouTubeResourceFinder()
    return youtube_finder


def init_youtube_finder():
    """Inicializar finder"""
    global youtube_finder
    youtube_finder = YouTubeResourceFinder()
    logger.info("YouTube Finder inicializado")
